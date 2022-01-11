# Generated by Django 3.2.10 on 2021-12-18 05:34

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('DressApp', '0005_alter_human_body_height'),
    ]

    operations = [
        migrations.CreateModel(
            name='Human_materials',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('cut_image_path', models.URLField(max_length=1000)),
                ('part_segm_path', models.URLField(max_length=1000)),
                ('skeleton_json_path', models.URLField(max_length=1000)),
                ('human_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='DressApp.human_body')),
            ],
        ),
    ]