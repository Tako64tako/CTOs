from .semantic_segmentation import exif_cut
from .semantic_segmentation import haikei
from .indexnet_matting.scripts import image_matting
from .indexnet_matting.scripts import cut
from .simple_bodypix_python import body_part_segm
from .pytorch_openpose import pose_check

import time
import json
import os
import cv2
import numpy as np

def image_cut(filename,input_img_path,height):
    imgProc_start = time.time()
    
    # リサイズして、フォーマット変換して、exif情報の処理をし、圧縮してdir_pathに保存する
    dir_path = "./DressApp/dress_lib/images/temporary_imgs/"
    filename = exif_cut.exifcut_compression_risize(filename,input_img_path,dir_path)

    # セマンティックセグメンテーション等の処理を行い、trimapとblur_imgを返す
    blur_img,trimap = haikei.cutting_out(dir_path,filename)
    #trimaps_dir = "./DressApp/dress_lib/images/trimaps/"
    #images_dir = "./DressApp/dress_lib/images/blur_images/"
    #cv2.imwrite(images_dir+filename,blur_img)#確認保存用
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

    # 骨格情報をJSONデータにする
    print(type(candidate))
    print(type(candidate[0]))
    print(type(candidate[0][0]))
    print(candidate)
    candidate_list = candidate.tolist()
    print(type(candidate_list))
    print(type(candidate_list[0]))
    print(type(candidate_list[0][0]))
    print(candidate_list)
    candidate_json = json.dumps(candidate_list)

    #os.remove('./DressApp/dress_lib/images/via/'+filename)#処理時に使った経由用の画像を消しておく
    imgProc_time = time.time() - imgProc_start
    print ("imgProc_time:{0}".format(imgProc_time) + "[sec]")
    return(filename,"imgProc_time:{0}".format(imgProc_time) + "[sec]",actual_img,human_segm,candidate_json)

'''if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread("./materials/part_segms/IMG_0137.png",0)
    plt.title('Segmentation Mask')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(img)
    plt.show()'''