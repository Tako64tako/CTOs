import os
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from .src import util
from .src.body import Body
 
POSE_SAVE_NAME = "img.png"
LOAD_MODEL_PATH = './DressApp/dress_lib/pytorch_openpose_master/model/body_pose_model.pth'
OUT_DIR_PATH = "./DressApp/dress_lib/images/skeleton_images/"

def scale_to_height(img,height):
    """高さが指定した値になるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))


    return dst

def pose_esti(actual_img,file_name):#acutual_img = BGRA
    body_estimation = Body(LOAD_MODEL_PATH)
    oriImg = cv2.cvtColor(actual_img,cv2.COLOR_BGRA2BGR)

    yohaku = 60
    h,w = oriImg.shape[:2]
    waku = np.full((h,w+(yohaku*2),3),255,np.uint8)
    waku_h,waku_w = waku.shape[:2]
    for y in range(waku_h):
        for x in range(waku_w):
            if x >= yohaku and x < w + yohaku:
                waku[y][x] = oriImg[y][x-yohaku]
    
    candidate, subset = body_estimation(waku)
    '''plt.imshow(waku[0:h,yohaku:waku_w-yohaku])
    plt.show()'''
    waku = waku[0:h,yohaku:waku_w-yohaku]#余白を削る
    '''plt.imshow(cv2.cvtColor(waku,cv2.COLOR_BGR2RGB))
    plt.show()'''

    for i in range(18):
        candidate[i][0] -= yohaku
    print("candidate")
    print(candidate)
    print("subset")
    print(subset)

    canvas = copy.deepcopy(waku)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    result_image_path = OUT_DIR_PATH + file_name
    cv2.imwrite(result_image_path, canvas)#確認保存用
    plt.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    plt.show()
    
    return(candidate)
