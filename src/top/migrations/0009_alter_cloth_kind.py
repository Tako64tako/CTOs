# Generated by Django 3.2.9 on 2021-11-12 07:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('top', '0008_cloth_img'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cloth',
            name='kind',
            field=models.CharField(max_length=80),
        ),
    ]
