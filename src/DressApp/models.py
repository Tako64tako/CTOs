from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
# このモデルは
# 人の名前
# 身長
# クライアントから位受け取った画像のパス(mediaからのパス)
# 背景削除画像のパス(mediaからのパス)と
# 人物部位セグメンテーション画像のパス(srcからdress_lib/dress_lib/materials/part_segms/画像ファイルまでのパス)
# 骨格情報を持つjsonファイルのパス(srcからdress_lib/dress_lib/materials/skeleton_jsons/jsonファイルまでのパス)
# といった入力されたものから画像処理を行なった情報までをdbに登録するためのもの
class Human_body(models.Model):
    # human_nameプロパティは最大100文字でユニークの文字列
    human_name = models.CharField(max_length=100)#, unique=True)#人の名前
    height = models.IntegerField(default=0,validators=[MinValueValidator(1), MaxValueValidator(250)])#身長
    picture = models.ImageField(upload_to="human_img/",blank=True,null=True)#クライアントから受け取った画像のパス(mediaからのパス)
    
    cut_image_path = models.URLField(default='none_path',max_length=1000)#背景削除画像のパス(mediaからのパス)
    part_segm_path = models.URLField(default='none_path',max_length=1000)#人物部位セグメンテーション画像のパス(srcからdress_lib/dress_lib/materials/part_segms/画像ファイルまでのパス)
    skeleton_json_path = models.URLField(default='none_path',max_length=1000)#骨格情報を持つjsonファイルのパス(srcからdress_lib/dress_lib/materials/skeleton_jsons/jsonファイルまでのパス)
    def __str__(self):
        return self.id

class Human_clothes(models.Model):
    human_id = models.IntegerField(default=0)
    part_clothes = models.IntegerField(default=0)
    clothes_name = models.CharField(max_length=100)
    match_clothes_path = models.URLField(default='none_path',max_length=1000)