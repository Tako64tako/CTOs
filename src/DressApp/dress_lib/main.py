from semantic_segmentation import exif_cut
#segmかhaikeiどちらか一方だけimportする 両方importするとoutofmemoryエラーになる
#import segm
from semantic_segmentation import haikei
from indexnet_matting.scripts import demo
from indexnet_matting.scripts import cut
from simple_bodypix_python_master import body_part_segm
from pytorch_openpose_master import pose_check
from change_clothes_lib import clothes_on_top

import time
import cv2
import matplotlib.pyplot as plt
import json

def image_cut():
    start = time.time()
    #exif_cut.exifcut_compression_risize("segm")#haikeiを実行するならコメントアウト
    #segm.cutting_out()#haikeiを実行するならコメントアウト
    dir_path,file_name = exif_cut.exifcut_compression_risize("haikei")#segmを実行するならコメントアウト
    blur_img,trimap,image_path = haikei.cutting_out(dir_path,file_name)#segmを実行するならコメントアウト
    matte = demo.infer(blur_img,trimap,image_path)
    alpha_img_path = cut.cutting(blur_img,matte,file_name)

    actual_img,human_segm = body_part_segm.segm_run(alpha_img_path,file_name)
    candidate = pose_check.pose_esti(actual_img,file_name)
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
    '''#JSONデータを保存
    with open('./DressApp/dress_lib/address.json', 'w') as fp:
        json.dump(candidate_json, fp, indent=4, ensure_ascii=False)'''

    plt.imshow(actual_img)
    plt.show()
    
    result_img = None
    part_clothes = 1
    clothes_name = "model_0"
    if part_clothes == 1:
        result_img = clothes_on_top.change(actual_img,human_segm,candidate,clothes_name)
    cv2.imwrite("./DressApp/dress_lib/images/result_images/"+file_name,result_img)
    
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

image_cut()