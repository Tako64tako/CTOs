from django.db.models.signals import pre_delete
from django.dispatch import receiver

from .models import Human_body,Human_clothes
import os


# dbのHuman_bodyテーブルのデータを消すときに画像も消しておく
@receiver(pre_delete, sender=Human_body)
def pre_human_body_delete(sender, instance, **kwargs):
    if os.path.isfile('.'+instance.cut_image_path)==True:
        os.remove('.'+instance.cut_image_path)#./media/cut_images/xxxを消す
    if os.path.isfile(instance.part_segm_path)==True:
        os.remove(instance.part_segm_path)#./DressApp/dress_lib/materials/part_segms/xxxを消す
    if os.path.isfile(instance.skeleton_json_path)==True:
        os.remove(instance.skeleton_json_path)#./DressApp/dress_lib/materials/skeleton_jsons/xxxを消す
    if os.path.isfile('./DressApp/dress_lib/images/via/'+str(instance.picture).split('/')[1])==True:
        os.remove('./DressApp/dress_lib/images/via/'+str(instance.picture).split('/')[1])#処理時に使った経由用の画像./DressApp/dress_lib/images/via/xxxを消す
    
    # 関連するHuman_clothesテーブルのデータも消しておく
    id = instance.id#human_bodyのidを取得
    human_clothes_list = Human_clothes.objects.all().filter(human_id=id)#human_bodyのidに関連したデータリストを取得
    human_clothes_list.delete()#そのリストの全てのデータをdbから消す
    print("human_bodyを削除しました")

# dbのHuman_clothesテーブルのデータを消すときに画像も消しておく
@receiver(pre_delete, sender=Human_clothes)
def pre_human_clothes_delete(sender, instance, **kwargs):
    if os.path.isfile(instance.match_clothes_path)==True:
        os.remove(instance.match_clothes_path)#./DressApp/dress_lib/materials/match_clothes/xxxを消す
    print("human_clothesを削除しました")