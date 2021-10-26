from django.db import models

class Cloth(models.Model):
    # 商品番号 : INTEGR型で、主キー
    number = models.CharField(max_length=100,primary_key=True)
    # 商品名 : 文字列30桁
    name = models.CharField(max_length=30)
    # 性別 : 文字列30桁
    gender = models.CharField(max_length=30)
    # 種類 : 文字列30桁
    kind = models.CharField(max_length=30)
    # サイズ : 文字列30桁
    size = models.CharField(max_length=30)
    # 色 : 文字列30桁
    color = models.CharField(max_length=30)
    #
    def __str__(self):
        return self.name;