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
import numpy as np

def image_cut(filename,input_img_path,height):
    imgProc_start = time.time()
    #exif_cut.exifcut_compression_risize("segm")#haikeiを実行するならコメントアウト
    #segm.cutting_out()#haikeiを実行するならコメントアウト
    dir_path,filename = exif_cut.exifcut_compression_risize("haikei",filename,input_img_path)#segmを実行するならコメントアウト
    blur_img,trimap,image_path = haikei.cutting_out(dir_path,filename)#segmを実行するならコメントアウト
    #trimaps_dir = "./DressApp/dress_lib/images/trimaps/"
    #images_dir = "./DressApp/dress_lib/images/images/"
    #cv2.imwrite(images_dir+filename,blur_img)#確認保存用
    #cv2.imwrite(trimaps_dir+filename,trimap)#確認保存用
    matte = demo.infer(blur_img,trimap,filename)
    #RESULT_DIR = './DressApp/dress_lib/images/mattes'
    #RESULT_DIR = "./DressApp/dress_lib/images/via/"#臨時追加
    #Image.fromarray(alpha.astype(np.uint8)).save(os.path.join(RESULT_DIR, filename))#確認保存用
    alpha_img = cut.cutting(blur_img,matte,filename)
    #cv2.imwrite("./DressApp/dress_lib/images/via/"+filename,alpha_img)#確認保存用

    actual_img,human_segm = body_part_segm.segm_run(alpha_img,height)
    #outdirPath = "./DressApp/dress_lib/images/part_segm_images/"
    #cv2.imwrite(outdirPath+file_name,human_segm)#確認保存用
    #cv2.imwrite("./DressApp/dress_lib/images/cut_images/",actualimg)
    candidate = pose_check.pose_esti(actual_img,filename)
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
    
    '''actual_img = cv2.imread("./media/cut_images/IMG_0137.png",-1)
    human_segm = cv2.imread("./DressApp/dress_lib/materials/part_segms/IMG_0137.png",0)
    with open("./DressApp/dress_lib/materials/skeleton_jsons/IMG_0137.json") as f:
        candidate_txt = json.load(f)#strになる
        print(type(candidate_txt))
        candidate = json.loads(candidate_txt)#listになる
        print(type(candidate))
        candidate = np.array(candidate)#ndarrayになる
        print(type(candidate))'''
    result_img = None
    part_clothes = 1
    clothes_name = "model_0"
    if part_clothes == 1:
        result_img,brank_img = clothes_on_top.change(actual_img,human_segm,candidate,clothes_name)
    cv2.imwrite("./DressApp/dress_lib/images/result_images/"+filename,result_img)
    
    imgProc_time = time.time() - imgProc_start
    print ("imgProc_time:{0}".format(imgProc_time) + "[sec]")

image_cut("IMG_0138.png","./media/human_img/IMG_0138.png",169)