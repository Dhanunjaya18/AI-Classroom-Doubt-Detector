from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('student/', views.student_dashboard, name='student_dashboard'),
    path('teacher/', views.teacher_dashboard, name='teacher_dashboard'),
    path('api/submit-doubt/', views.submit_doubt, name='submit_doubt'),
    path('api/upvote/<int:doubt_id>/', views.upvote_doubt, name='upvote_doubt'),
    path('api/search/', views.search_doubts, name='search_doubts'),
    path('api/feed/', views.get_doubts_feed, name='doubts_feed'),
    path('api/cluster/', views.trigger_clustering, name='trigger_clustering'),
    path('api/all-doubts/', views.all_doubts, name='all_doubts'),
]
