from semantic_segmentation import exif_cut
from semantic_segmentation import haikei
from indexnet_matting.scripts import image_matting
from indexnet_matting.scripts import cut
from simple_bodypix_python import body_part_segm
from pytorch_openpose import pose_check
from change_clothes_lib import clothes_on_top

import time
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np

# 入力画像、身長から着せ替えを行う
def segm_dress(filename,input_img_path,height):
    imgProc_start = time.time()
    
    # リサイズして、フォーマット変換して、exif情報の処理をし、圧縮してdir_pathに保存する
    dir_path = "./DressApp/dress_lib/images/temporary_imgs/"
    filename = exif_cut.exifcut_compression_risize(filename,input_img_path,dir_path)

    # セマンティックセグメンテーション等の処理を行い、trimapとblur_imgを返す
    blur_img,trimap = haikei.cutting_out(dir_path,filename)
    #trimaps_dir = "./DressApp/dress_lib/images/trimaps/"
    images_dir = "./DressApp/dress_lib/images/blur_images/"
    cv2.imwrite(images_dir+filename,blur_img)#確認保存用
    #cv2.imwrite(trimaps_dir+filename,trimap)#確認保存用

    # indexnet_mattingで背景を綺麗に切り抜りとったマスクを得る
    matte = image_matting.infer(blur_img,trimap,filename)
    #RESULT_DIR = './DressApp/dress_lib/images/mattes'
    #Image.fromarray(alpha.astype(np.uint8)).save(os.path.join(RESULT_DIR, filename))#確認保存用

    # matteをもとにblur_imgの背景を切り取ることで背景削除人物画像を得る
    alpha_img = cut.cutting(blur_img,matte,filename)
    #cut_dir='./DressApp/dress_lib/images/cut_images/'
    #cv2.imwrite(cut_dir+filename,alpha_img)#確認保存用

    # alpha_imgとheightから人物の身長を合わせてリサイズした画像を生成し、bodypixで人物の部位ごとにセグメンテーションした画像を得る
    actual_img,human_segm = body_part_segm.segm_run(alpha_img,height)
    #outdirPath = "./DressApp/dress_lib/images/part_segm_images/"
    #height_resize_dir = "./DressApp/dress_lib/images/height_resize_images/"
    #cv2.imwrite(outdirPath+filename,human_segm)#確認保存用
    #cv2.imwrite(height_resize_dir+filename,actual_img)

    # openposeで人物の骨格情報を得る
    candidate,canvas = pose_check.pose_esti(actual_img,filename)
    #skeleton_dir = "./DressApp/dress_lib/images/skeleton_images/"
    #cv2.imwrite(skeleton_dir+filename,canvas)#確認保存用

    #背景削除人物画像を保存
    actual_path = "./DressApp/dress_lib/materials/actual_images/"+filename
    cv2.imwrite(actual_path,actual_img)

    #人物部位セグメンテーション画像を保存
    human_segm_path = "./DressApp/dress_lib/materials/part_segms/"+filename
    cv2.imwrite(human_segm_path,human_segm)

    #骨格情報をJSONデータにして保存
    candidate_list = candidate.tolist()
    candidate_json = json.dumps(candidate_list)
    skeleton_path='./DressApp/dress_lib/materials/skeleton_jsons/'+filename.split('.')[0]+".json"
    with open(skeleton_path, 'w') as fp:
        json.dump(candidate_json, fp, indent=4, ensure_ascii=False)

    
    # 着せ替える部位(part_clothes)と着せ替える服の名称を決めて着せ替えを行う
    part_clothes = 1
    clothes_name = "model_4"
    if part_clothes == 1:
        result_img,brank_img = clothes_on_top.change(actual_img,human_segm,candidate,clothes_name)
    else:
        return

    #加工した服画像を保存
    material_brank_dir = "./DressApp/dress_lib/materials/match_clothes/"
    cv2.imwrite(material_brank_dir+str(part_clothes)+"_"+clothes_name+"_"+filename,brank_img)

    #着せ替え画像を保存
    result_dir = "./DressApp/dress_lib/images/result_images/"
    cv2.imwrite(result_dir+filename,result_img)#確認保存用
    
    imgProc_time = time.time() - imgProc_start
    print ("imgProc_time:{0}".format(imgProc_time) + "[sec]")

segm_dress("IMG_0137.png","./media/human_img/IMG_0137.png",178)