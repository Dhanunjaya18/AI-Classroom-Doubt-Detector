from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, Doubt, Vote

@admin.register(User)
class CustomUserAdmin(UserAdmin):
    list_display = ['username', 'email', 'role', 'is_active']
    list_filter = ['role']
    fieldsets = UserAdmin.fieldsets + (('Profile', {'fields': ('role', 'avatar_color')}),)

@admin.register(Doubt)
class DoubtAdmin(admin.ModelAdmin):
    list_display = ['text', 'student', 'subject', 'cluster_id', 'upvotes', 'timestamp']
    list_filter = ['subject', 'cluster_id']

@admin.register(Vote)
class VoteAdmin(admin.ModelAdmin):
    list_display = ['student', 'doubt', 'timestamp']
