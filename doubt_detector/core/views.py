from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.db.models import Count
from django.utils import timezone
from datetime import timedelta
import json
import random

from .models import User, Doubt, Vote
from .ml_pipeline import run_clustering, find_similar_doubts, detect_subject, get_top_keywords

AVATAR_COLORS = ['#4F46E5', '#7C3AED', '#2563EB', '#0891B2', '#059669', '#D97706', '#DC2626', '#DB2777']


def index(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return redirect('login')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('dashboard')
        messages.error(request, 'Invalid username or password.')
    return render(request, 'core/login.html')


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        role = request.POST.get('role', 'student')
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
        elif len(password) < 4:
            messages.error(request, 'Password must be at least 4 characters.')
        else:
            user = User.objects.create_user(
                username=username, email=email, password=password,
                role=role, avatar_color=random.choice(AVATAR_COLORS)
            )
            login(request, user)
            return redirect('dashboard')
    return render(request, 'core/signup.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def dashboard(request):
    if request.user.is_teacher():
        return redirect('teacher_dashboard')
    return redirect('student_dashboard')


@login_required
def student_dashboard(request):
    if not request.user.is_student():
        return redirect('teacher_dashboard')
    
    doubts = Doubt.objects.select_related('student').all()
    my_doubts = doubts.filter(student=request.user)
    recent_doubts = doubts[:10]
    voted_ids = list(Vote.objects.filter(
        student=request.user
    ).values_list('doubt_id', flat=True))

    context = {
        'my_doubts': my_doubts,
        'recent_doubts': recent_doubts,
        'total_doubts': doubts.count(),
        'my_count': my_doubts.count(),
        'voted_ids': json.dumps(voted_ids),
    }
    return render(request, 'core/student_dashboard.html', context)


@login_required
def teacher_dashboard(request):
    if not request.user.is_teacher():
        return redirect('student_dashboard')
    
    doubts = Doubt.objects.select_related('student').all()
    total = doubts.count()
    
    # Subject distribution
    subject_data = {}
    for d in doubts:
        subj = d.subject or detect_subject(d.text)
        subject_data[subj] = subject_data.get(subj, 0) + 1
    
    # Cluster data
    cluster_data = {}
    for d in doubts.exclude(cluster_id=None):
        label = d.cluster_label or f'Cluster {d.cluster_id}'
        cluster_data[label] = cluster_data.get(label, 0) + 1
    
    # Timeline (last 7 days)
    timeline = []
    for i in range(6, -1, -1):
        day = timezone.now().date() - timedelta(days=i)
        count = doubts.filter(timestamp__date=day).count()
        timeline.append({'date': str(day), 'count': count})
    
    # Top doubts by votes
    top_doubts = doubts.order_by('-upvotes')[:5]
    
    # Recent doubts
    recent = doubts[:15]
    
    # Trending subjects (most doubts today)
    today_doubts = doubts.filter(timestamp__date=timezone.now().date())
    trending = {}
    for d in today_doubts:
        subj = d.subject or 'General'
        trending[subj] = trending.get(subj, 0) + 1
    trending_topics = sorted(trending.items(), key=lambda x: x[1], reverse=True)[:3]
    
    # Last clustering stats
    clustered = doubts.exclude(cluster_id=None).count()
    
    context = {
        'total_doubts': total,
        'total_students': User.objects.filter(role='student').count(),
        'clustered_doubts': clustered,
        'subject_data_json': json.dumps(subject_data),
        'cluster_data_json': json.dumps(cluster_data),
        'timeline_json': json.dumps(timeline),
        'top_doubts': top_doubts,
        'recent_doubts': recent,
        'trending_topics': trending_topics,
        'doubts_today': today_doubts.count(),
    }
    return render(request, 'core/teacher_dashboard.html', context)


@login_required
@require_POST
def submit_doubt(request):
    text = request.POST.get('text', '').strip()
    subject = request.POST.get('subject', '').strip()
    
    if not text or len(text) < 10:
        return JsonResponse({'status': 'error', 'message': 'Doubt must be at least 10 characters.'})
    
    # Auto-detect subject if not provided
    if not subject:
        subject = detect_subject(text)
    
    doubt = Doubt.objects.create(
        text=text, student=request.user, subject=subject
    )
    
    # Find similar doubts
    other_doubts = Doubt.objects.exclude(id=doubt.id)
    similar = find_similar_doubts(text, other_doubts, top_n=3)
    
    # Re-run clustering if enough doubts
    all_doubts = Doubt.objects.all()
    cluster_result = None
    if all_doubts.count() >= 2:
        cluster_result = run_clustering(all_doubts)
        doubt.refresh_from_db()
    
    similar_data = [
        {
            'id': s['doubt'].id,
            'text': s['doubt'].text[:120],
            'similarity': int(s['similarity'] * 100),
            'student': s['doubt'].student.username,
            'subject': s['doubt'].subject,
        }
        for s in similar
    ]
    
    return JsonResponse({
        'status': 'success',
        'doubt_id': doubt.id,
        'subject': doubt.subject,
        'similar': similar_data,
        'cluster_label': doubt.cluster_label or '',
        'message': 'Doubt submitted successfully!'
    })


@login_required
@require_POST
def upvote_doubt(request, doubt_id):
    doubt = get_object_or_404(Doubt, id=doubt_id)
    vote, created = Vote.objects.get_or_create(student=request.user, doubt=doubt)
    
    if not created:
        vote.delete()
        doubt.upvotes = max(0, doubt.upvotes - 1)
        doubt.save()
        return JsonResponse({'status': 'removed', 'upvotes': doubt.upvotes})
    
    doubt.upvotes += 1
    doubt.save()
    return JsonResponse({'status': 'added', 'upvotes': doubt.upvotes})


@login_required
def search_doubts(request):
    query = request.GET.get('q', '').strip()
    doubts = []
    if query:
        all_doubts = Doubt.objects.exclude(student=request.user)
        results = find_similar_doubts(query, all_doubts, top_n=10)
        doubts = [
            {
                'id': r['doubt'].id,
                'text': r['doubt'].text,
                'similarity': int(r['similarity'] * 100),
                'student': r['doubt'].student.username,
                'subject': r['doubt'].subject or 'General',
                'upvotes': r['doubt'].upvotes,
                'time': r['doubt'].timestamp.strftime('%b %d, %Y'),
            }
            for r in results
        ]
    return JsonResponse({'results': doubts, 'query': query})


@login_required
def get_doubts_feed(request):
    page = int(request.GET.get('page', 1))
    per_page = 10
    offset = (page - 1) * per_page
    
    doubts = Doubt.objects.select_related('student').all()[offset:offset + per_page]
    voted_ids = set(Vote.objects.filter(student=request.user).values_list('doubt_id', flat=True))
    
    data = [
        {
            'id': d.id,
            'text': d.text,
            'student': d.student.username,
            'avatar_color': d.student.avatar_color,
            'subject': d.subject or 'General',
            'upvotes': d.upvotes,
            'voted': d.id in voted_ids,
            'cluster_label': d.cluster_label or '',
            'time': d.timestamp.strftime('%b %d, %Y %H:%M'),
        }
        for d in doubts
    ]
    return JsonResponse({'doubts': data, 'has_more': Doubt.objects.count() > offset + per_page})


@login_required
def trigger_clustering(request):
    if not request.user.is_teacher():
        return JsonResponse({'status': 'error', 'message': 'Permission denied'})
    
    all_doubts = Doubt.objects.all()
    if all_doubts.count() < 2:
        return JsonResponse({'status': 'error', 'message': 'Need at least 2 doubts to cluster'})
    
    result = run_clustering(all_doubts)
    return JsonResponse(result)


@login_required
def all_doubts(request):
    if not request.user.is_teacher():
        return JsonResponse({'status': 'error'})
    
    subject_filter = request.GET.get('subject', '')
    cluster_filter = request.GET.get('cluster', '')
    
    doubts = Doubt.objects.select_related('student').all()
    if subject_filter:
        doubts = doubts.filter(subject=subject_filter)
    if cluster_filter:
        doubts = doubts.filter(cluster_label__icontains=cluster_filter)
    
    data = [
        {
            'id': d.id,
            'text': d.text,
            'student': d.student.username,
            'subject': d.subject or 'General',
            'cluster': d.cluster_label or 'Unclustered',
            'upvotes': d.upvotes,
            'time': d.timestamp.strftime('%b %d, %Y %H:%M'),
        }
        for d in doubts
    ]
    return JsonResponse({'doubts': data})
