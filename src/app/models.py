from django.db import models

class Area(models.Model):
    name = models.CharField(max_length=100)
    create_date = models.DateTimeField('date published')

    def __str__(self):
        return self.name;

class Cafe(models.Model):
    area = models.ForeignKey(Area, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    memo = models.CharField(max_length=400)
    website = models.URLField()
    image_path = models.CharField(max_length=400)
    create_date = models.DateTimeField('date published')

    def __str__(self):
        return self.name;

class Utility(models.Model):
    key = models.CharField(max_length=100)
    value = models.CharField(max_length=100)
    create_date = models.DateTimeField('date published')

    def __str__(self):
        return self.key;