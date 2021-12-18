from .semantic_segmentation import exif_cut
#segmかhaikeiどちらか一方だけimportする 両方importするとoutofmemoryエラーになる
#from . import segm
from .semantic_segmentation import haikei
from .indexnet_matting.scripts import demo
from .indexnet_matting.scripts import cut
from .simple_bodypix_python_master import body_part_segm
from .pytorch_openpose_master import pose_check

import time
import json

def image_cut(filename):
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

    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    return("elapsed_time:{0}".format(elapsed_time) + "[sec]",actual_img,human_segm,candidate_json)