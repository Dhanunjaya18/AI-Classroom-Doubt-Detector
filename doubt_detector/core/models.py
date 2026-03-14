from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    ROLE_CHOICES = [('student', 'Student'), ('teacher', 'Teacher')]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='student')
    avatar_color = models.CharField(max_length=7, default='#4F46E5')

    def is_teacher(self):
        return self.role == 'teacher'

    def is_student(self):
        return self.role == 'student'


class Doubt(models.Model):
    text = models.TextField()
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='doubts')
    timestamp = models.DateTimeField(auto_now_add=True)
    cluster_id = models.IntegerField(null=True, blank=True)
    cluster_label = models.CharField(max_length=200, blank=True)
    subject = models.CharField(max_length=100, blank=True)
    upvotes = models.IntegerField(default=0)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return self.text[:60]


class Vote(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE, related_name='votes')
    doubt = models.ForeignKey(Doubt, on_delete=models.CASCADE, related_name='vote_set')
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('student', 'doubt')
