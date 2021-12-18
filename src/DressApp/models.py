from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator

# Create your models here.
#このモデルは人の名前と身長とクライアントから受け取った画像のパス(mediaからのパス)をdbに登録するためのもの
class Human_body(models.Model):
    # human_nameプロパティは最大100文字でユニークの文字列
    id = models.AutoField(primary_key=True)#id

    human_name = models.CharField(max_length=100)#, unique=True)#人の名前
    
    height = models.IntegerField(default=0,validators=[MinValueValidator(1), MaxValueValidator(250)])#身長
    
    picture = models.ImageField(upload_to="human_img/",blank=True,null=True)#クライアントから受け取った画像のパス(mediaからのパス)
    def __str__(self):
        return self.id

# このモデルは
# 背景削除画像のパス(mediaからのパス)と
# 人物部位セグメンテーション画像のパス(srcからdress_lib/dress_lib/materials/part_segms/画像ファイルまでのパス)
# 骨格情報を持つjsonファイルのパス(srcからdress_lib/dress_lib/materials/skeleton_jsons/jsonファイルまでのパス)
# といった入力された画像から画像処理を行なった情報をdbに登録するためのもの
# Human_bodyと一対一の関係にしたいからhuman_idで外部キーつけてみた
class Human_materials(models.Model):
    id = models.AutoField(primary_key=True)
    cut_image_path = models.URLField(max_length=1000)#背景削除画像のパス(mediaからのパス)
    part_segm_path = models.URLField(max_length=1000)#人物部位セグメンテーション画像のパス(srcからdress_lib/dress_lib/materials/part_segms/画像ファイルまでのパス)
    skeleton_json_path = models.URLField(max_length=1000)#骨格情報を持つjsonファイルのパス(srcからdress_lib/dress_lib/materials/skeleton_jsons/jsonファイルまでのパス)
    human_id = models.ForeignKey(Human_body, on_delete=models.CASCADE)# Human_bodyと一対一の関係にしたいからhuman_idで外部キーつけてみた
