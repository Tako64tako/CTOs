# Generated by Django 3.2.9 on 2021-11-22 03:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('top', '0005_auto_20211122_1119'),
    ]

    operations = [
        migrations.AlterField(
            model_name='cloth',
            name='pictures_detail',
            field=models.URLField(max_length=1000),
        ),
    ]