from django.db import models
import django_cleanup

# Create your models here.
class UserProfile(models.Model):
    avator = models.ImageField(
        verbose_name = 'avator',
        upload_to = 'avator_images/'
    )