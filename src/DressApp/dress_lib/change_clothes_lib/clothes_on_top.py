import os
from re import search
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
import cv2

import json
from numpy import linalg as LA

CLOTHES_DIR = "./DressApp/dress_lib/clothes_datas/clothes_on_top/"

# 二つのベクトルからなす角を求めて返す。(例えば、人物の骨格情報から得た肩ベクトルと服の骨格情報から得た肩ベクトルとのなす角を求める)
def get_angleFrom2Vec(v,u):
    inner_product = np.dot(v, u)
    n = LA.norm(v) * LA.norm(u)#ベクトルの長さ同士の積
    coscos = inner_product / n
    formed_angle = np.rad2deg(np.arccos(np.clip(coscos, -1.0, 1.0)))
    return(formed_angle)

# ベクトルの内積の角度の正負を決める
def calc_rotation_direction(clothes_vec,human_vec,angle):#不完全
    c = 1.0
    if angle+c > 180:
        c = (180 - angle)/2
    
    sinsin = math.sin(math.radians(c))
    coscos = math.cos(math.radians(c))
    rot_x = (clothes_vec[0] * sinsin) - (clothes_vec[1] * sinsin)
    rot_y = (clothes_vec[0] * sinsin) + (clothes_vec[1] * coscos)
    com_cl_vec = np.array([rot_x,rot_y])
    com_angle = get_angleFrom2Vec(com_cl_vec,human_vec)
    check_back = 1
    if com_angle > angle:
        check_back = -1
    
    return(check_back)
            


# 人物部位セグメンテーション画像の襟の座標を取得して返す
def search_neck(human_segm):
    h,w = human_segm.shape[:2]
    search_kernel = [[-1,0],[0,-1],[1,0],[0,1]]
    neck_max_width_point = [0,0]
    neck_min_width_point = [w,0]
    '''plt.imshow(human_segm)
    plt.show()'''
    for y in range(h):
        for x in range(w):
            for n_list in search_kernel:
                karas_y = y+n_list[1]
                karas_x = x+n_list[0]
                if human_segm[y][x] != 8:
                    break

                elif karas_y>=0 and karas_y<h and karas_x>=0 and karas_x<w:#out of index error対策
                    if human_segm[karas_y][karas_x] == 1:
                        if neck_max_width_point[0] < karas_x:
                            neck_max_width_point = [karas_x,karas_y]

                        if neck_min_width_point[0] > karas_x:
                            neck_min_width_point = [karas_x,karas_y]
    
    print("neck_max_width_point="+str(neck_max_width_point))
    print("neck_min_width_point="+str(neck_min_width_point))
    return(neck_max_width_point,neck_min_width_point)

# 体部分の上の服の襟に相当する座標４点をプロットした画像を
# json_dataのright_point,left_point,heightから作成する。
def create_neckPointMask(torso_img,json_data):
    h,w = torso_img.shape[:2]
    neck_point_mask = np.zeros((h,w,1),np.uint8)
    right_point_down = [json_data["neck"]["right_point"][0],json_data["neck"]["height"][1]]
    left_point_down = [json_data["neck"]["left_point"][0],json_data["neck"]["height"][1]]

    neck_point_mask[json_data["neck"]["right_point"][1]][json_data["neck"]["right_point"][0]] = 255
    neck_point_mask[json_data["neck"]["left_point"][1]][json_data["neck"]["left_point"][0]] = 255
    neck_point_mask[right_point_down[1]][right_point_down[0]] = 255
    neck_point_mask[left_point_down[1]][left_point_down[0]] = 255

    neck_point_mask = cv2.dilate(neck_point_mask,(5,5),iterations = 3)
    '''plt.title("neck_point_mask")
    plt.imshow(neck_point_mask)
    plt.show()'''
    return(neck_point_mask)

# 服(体)画像の横縦幅をh+wの正方形にした画像を作成する※回転した際に服(体)自体が画像からはみ出ないようにするため
# その服(体)画像を肩なす角だけ回転させる
# 回転させた服(体)画像をバウンディングボックスサイズで切り取る
# 人物部位セグム画像を胴体部分のバウンディングボックスサイズを取得する
# 服の体画像を(人物部位セグム画像の胴体の高さ, 服の襟座標を人物部位セグム画像の襟座標に合わさるような横幅)にリサイズする
def adjust_torso_rotate(actual_img,result_img,human_segm,torso_img,neck_point_mask,shoulder_formed_angle,json_data,neck_max_width_point,neck_min_width_point):
    #１ 服(体)画像の横縦幅をh+wの正方形にした画像を作成する※回転した際に服(体)自体が画像からはみ出ないようにするため
    h,w = torso_img.shape[:2]
    h_l = h+w
    w_l = h+w
    h_l_sa = h_l - h
    w_l_sa = w_l - w
    if h_l_sa%2 == 1:
        h_l += 1
        h_l_sa = h_l - h
    if w_l_sa%2 == 1:
        w_l += 1
        w_l_sa = w_l - w
    
    sy = int(h_l_sa/2)
    sx = int(w_l_sa/2)
    square_torso = np.zeros((h_l,w_l,4),np.uint8)
    square_neck_point_mask = np.zeros((h_l,w_l,1),np.uint8)
    for y in range(h_l):
        for x in range(w_l):
            if y>=sy and y<h+sy and x>=sx and x<w+sx:
                square_torso[y][x] = torso_img[y - sy][x - sx]
                square_neck_point_mask[y][x] = neck_point_mask[y - sy][x - sx]
    

    #２ その服(体)画像を肩なす角だけ回転させる
    #回転の中心を指定  
    center = (int(w_l/2), int(h_l/2))
    #スケールを指定
    scale = 1.0
    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, shoulder_formed_angle , scale)
    #アフィン変換
    square_torso = cv2.warpAffine(square_torso, trans, (w_l,h_l))
    square_neck_point_mask = cv2.warpAffine(square_neck_point_mask, trans, (w_l,h_l))

    '''plt.title("afin")
    plt.imshow(square_torso)
    plt.show()
    plt.title("afin_neck")
    plt.imshow(square_neck_point_mask)
    plt.show()'''


    #３ 回転させた服(体)画像をバウンディングボックスサイズで切り取る
    torso_mask = np.zeros((h_l,w_l,1),np.uint8)
    max_y = 0#服画像の服領域の最大y
    min_y = h_l
    max_x = 0
    min_x = w_l
    for y in range(h_l):
        for x in range(w_l):
            if square_torso[y][x][3] != 0:
                torso_mask[y][x] = 255
                if max_y < y:
                    max_y = y
                if min_y > y:
                    min_y = y
                if max_x < x:
                    max_x = x
                if min_x > x:
                    min_x = x
    print("min_y="+str(min_y))
    print("max_y="+str(max_y))
    print("min_x="+str(min_x))
    print("max_x="+str(max_x))
    print("y="+str(max_y-min_y))
    print("x="+str(max_x-min_x))
    '''plt.title("torso_mask")
    plt.imshow(torso_mask[min_y:max_y,min_x:max_x])
    plt.show()'''


    #４ 人物部位セグム画像を胴体部分のバウンディングボックスサイズを取得する
    human_segm_h,human_segm_w = human_segm.shape[:2]
    human_segm_max_y = 0#服画像の服領域の最大y
    human_segm_min_y = human_segm_h
    human_segm_max_x = 0
    human_segm_min_x = human_segm_w
    for y in range(human_segm_h):
        for x in range(human_segm_w):
            if human_segm[y][x] == 8:
                if human_segm_max_y < y:
                    human_segm_max_y = y+1
                if human_segm_min_y > y:
                    human_segm_min_y = y
                if human_segm_max_x < x:
                    human_segm_max_x = x+1
                if human_segm_min_x > x:
                    human_segm_min_x = x
    print("human_segm_min_y="+str(human_segm_min_y))
    print("human_segm_max_y="+str(human_segm_max_y))
    print("human_segm_min_x="+str(human_segm_min_x))
    print("human_segm_max_x="+str(human_segm_max_x))
    print("y="+str(human_segm_max_y-human_segm_min_y))
    print("x="+str(human_segm_max_x-human_segm_min_x))
    '''plt.title("human_segm")
    plt.imshow(human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x])
    plt.show()'''
    human_torso = human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#人体部位セグメンテーションの胴体部分
    #result_img = actual_img.copy()#actual_imgをコピーする。これに服を着せる
    #print("result_img.shape="+str(result_img.shape))
    actual_torso = result_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の胴体部分
    human_torso_y,human_torso_x = human_torso.shape[:2]


    #５ 服の体画像を(人物部位セグム画像の胴体の高さ, 服の襟座標を人物部位セグム画像の襟座標に合わさるような横幅)にリサイズする
    resize_y = human_segm_max_y - human_segm_min_y
    resize_x = int((neck_max_width_point[0]-neck_min_width_point[0])/(json_data["neck"]["left_point"][0]-json_data["neck"]["right_point"][0])*(max_x-min_x))
    torso_img = square_torso[min_y:max_y,min_x:max_x].copy()
    torso_img = cv2.resize(torso_img, dsize=(resize_x, resize_y))#resizeした服の画像でこれを着せる
    neck_point_mask = square_neck_point_mask[min_y:max_y,min_x:max_x].copy()
    neck_point_mask = cv2.resize(neck_point_mask, dsize=(resize_x, resize_y))
    '''plt.title("resize_torso_img")
    plt.imshow(torso_img)
    plt.show()
    plt.title("resize_neck_point_mask")
    plt.imshow(neck_point_mask)
    plt.show()'''
    
    clothes_min_x = resize_x
    clothes_max_x = 0
    human_min_x = human_segm_max_x
    human_max_x = 0
    for y in range(resize_y):
        clothes_min_x = resize_x#リセットする
        clothes_max_x = 0
        human_min_x = human_segm_max_x
        human_max_x = 0
        print(str(y)+"回")
        for x in range(resize_x):
            if torso_img[y][x][3] != 0:
                if clothes_max_x < x:
                    clothes_max_x = x+1
                if clothes_min_x > x:
                    clothes_min_x = x
        print("clothes_max_x="+str(clothes_max_x))
        print("clothes_min_x="+str(clothes_min_x))
        for foolx in range(human_torso_x):
            if human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x][y][foolx] == 8:
                if human_max_x < foolx:
                    human_max_x = foolx+1
                if human_min_x > foolx:
                    human_min_x = foolx
        print("human_max_x="+str(human_max_x))
        print("human_min_x="+str(human_min_x))
        brank = torso_img[y:y+1,clothes_min_x:clothes_max_x]
        print(brank)

        brank = cv2.resize(brank,dsize=((human_max_x-human_min_x), 1))#ここで人物部位セグム画像の胴体xサイズにリサイズ
        print(brank)

        print(actual_torso.shape)
        for i in range(human_max_x-human_min_x):#ここで人物部位セグム画像の胴体xサイズにリサイズした画像をresult_imgに貼り付け
            if brank[0][i][3] != 0:
                actual_torso[y][human_min_x+i] = brank[0][i]
    '''plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()'''

    return(result_img)


def torso_change(actual_img,result_img,human_segm,candidate,model_name,json_data):
    torso_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "torso_"+model_name+".png",-1)

    right_sholder_point = np.array([int(candidate[2][0]),int(candidate[2][1])])
    left_sholder_point = np.array([int(candidate[5][0]),int(candidate[5][1])])
    shoulder_vec = left_sholder_point - right_sholder_point
    clothes_right_sholder_point = np.array(json_data["right_shoulder"])
    clothes_left_sholder_point = np.array(json_data["left_shoulder"])
    clothes_shoulder_vec = clothes_left_sholder_point - clothes_right_sholder_point
    shoulder_formed_angle = get_angleFrom2Vec(shoulder_vec,clothes_shoulder_vec)
    print(shoulder_formed_angle)

    neck_max_width_point,neck_min_width_point = search_neck(human_segm)#人物の襟の座標を取得

    neck_point_mask = create_neckPointMask(torso_img,json_data)#服の襟の範囲をプロットした二値画像を作成

    result_img = adjust_torso_rotate(actual_img,result_img,human_segm,torso_img,neck_point_mask,shoulder_formed_angle,json_data,neck_max_width_point,neck_min_width_point)
    return(result_img)


# 全体服画像の腕部分をlowerとupperを二等分する線の式を作成する
def create_bisector(right_arm_upper_vec,right_arm_lower_vec,right_shoulder_point,right_elbow_point,right_wrist_point):
    arm_angle = get_angleFrom2Vec(right_arm_upper_vec,right_arm_lower_vec) / 2
    print(arm_angle)
    if right_elbow_point[0] < right_wrist_point[0]:
        sinsin = math.sin(math.radians(-arm_angle))
        coscos = math.cos(math.radians(-arm_angle))
        rot_x = (right_arm_upper_vec[0] * sinsin) - (right_arm_upper_vec[1] * sinsin)
        rot_y = (right_arm_upper_vec[0] * sinsin) + (right_arm_upper_vec[1] * coscos)
    else:
        sinsin = math.sin(math.radians(arm_angle))
        coscos = math.cos(math.radians(arm_angle))
        rot_x = (right_arm_upper_vec[0] * sinsin) - (right_arm_upper_vec[1] * sinsin)
        rot_y = (right_arm_upper_vec[0] * sinsin) + (right_arm_upper_vec[1] * coscos)
    
    line_point = np.array([rot_x,rot_y]) #+ np.array(right_elbow_point)
    print("line_point="+str(line_point))
    a = (-line_point[1])/(-line_point[0]) # a = (Px1-Px2)/(Py1-Py2)
    #b = 0 - (a * 0)
    print("a="+str(a))
    #print("b="+str(b))
    return(a)

# 服(腕)画像、front服(腕)画像、back服(腕)画像をlower,upperを分けた画像をそれぞれ生成する
def upper_lower_split(right_arm_img,front_right_arm_img,back_right_arm_img,a,right_arm_img_elbow):
    '''plt.title("right_arm_img")
    plt.imshow(right_arm_img)
    plt.show()'''
    h,w = right_arm_img.shape[:2]
    upper_right_arm_img = np.zeros((h,w,4),np.uint8)
    lower_right_arm_img = np.zeros((h,w,4),np.uint8)
    upper_front_right_arm_img = np.zeros((h,w,4),np.uint8)
    lower_front_right_arm_img = np.zeros((h,w,4),np.uint8)
    upper_back_right_arm_img = np.zeros((h,w,4),np.uint8)
    lower_back_right_arm_img = np.zeros((h,w,4),np.uint8)
    
    for y in range(h):
        for x in range(w):
            if right_arm_img[y][x][3] == 255:
                if (a*(x-right_arm_img_elbow[0])+right_arm_img_elbow[1])<y:
                    lower_right_arm_img[y][x][3] = 255
                else:
                    upper_right_arm_img[y][x][3] = 255


            if front_right_arm_img[y][x][3] == 255:
                if (a*(x-right_arm_img_elbow[0])+right_arm_img_elbow[1])<y:
                    lower_front_right_arm_img[y][x][3] = 255
                else:
                    upper_front_right_arm_img[y][x][3] = 255

            if back_right_arm_img[y][x][3] == 255:
                if (a*(x-right_arm_img_elbow[0])+right_arm_img_elbow[1])<y:
                    lower_back_right_arm_img[y][x][3] = 255
                else:
                    upper_back_right_arm_img[y][x][3] = 255
    
    for y in range(h):
        for x in range(w):
            if right_arm_img[y][x][3] == 255 and upper_right_arm_img[y][x][3] == 255:
                upper_right_arm_img[y][x] = right_arm_img[y][x]
            elif right_arm_img[y][x][3] == 255 and lower_right_arm_img[y][x][3] == 255:
                lower_right_arm_img[y][x] = right_arm_img[y][x]
            
            if front_right_arm_img[y][x][3] == 255 and upper_front_right_arm_img[y][x][3] == 255:
                upper_front_right_arm_img[y][x] = front_right_arm_img[y][x]
            elif front_right_arm_img[y][x][3] == 255 and lower_front_right_arm_img[y][x][3] == 255:
                lower_front_right_arm_img[y][x] = front_right_arm_img[y][x]

            if back_right_arm_img[y][x][3] == 255 and upper_back_right_arm_img[y][x][3] == 255:
                upper_back_right_arm_img[y][x] = back_right_arm_img[y][x]
            elif back_right_arm_img[y][x][3] == 255 and lower_back_right_arm_img[y][x][3] == 255:
                lower_back_right_arm_img[y][x] = back_right_arm_img[y][x]
            

    '''plt.title("upper_right_arm_img")
    plt.imshow(upper_right_arm_img)
    plt.show()
    plt.title("lower_right_arm_img")
    plt.imshow(lower_right_arm_img)
    plt.show()
    plt.title("upper_front_right_arm_img")
    plt.imshow(upper_front_right_arm_img)
    plt.show()
    plt.title("lower_front_right_arm_img")
    plt.imshow(lower_front_right_arm_img)
    plt.show()
    plt.title("upper_back_right_arm_img")
    plt.imshow(upper_back_right_arm_img)
    plt.show()
    plt.title("lower_back_right_arm_img")
    plt.imshow(lower_back_right_arm_img)
    plt.show()'''

    return(upper_right_arm_img,lower_right_arm_img,upper_front_right_arm_img,lower_front_right_arm_img,upper_back_right_arm_img,lower_back_right_arm_img)

def adjust_upper_arm_rotate(actual_img,result_img,human_segm,upper_right_arm_img,upper_front_right_arm_img,upper_back_right_arm_img,right_arm_upper_angle,seihu,part_value):
    #１ 服(上腕)画像の横縦幅をh+wの正方形にした画像を作成する※回転した際に服(上腕)自体が画像からはみ出ないようにするため
    h,w = upper_right_arm_img.shape[:2]
    h_l = h+w
    w_l = h+w
    h_l_sa = h_l - h
    w_l_sa = w_l - w
    if h_l_sa%2 == 1:
        h_l += 1
        h_l_sa = h_l - h
    if w_l_sa%2 == 1:
        w_l += 1
        w_l_sa = w_l - w
    
    sy = int(h_l_sa/2)
    sx = int(w_l_sa/2)
    square_upper_right_arm_img = np.zeros((h_l,w_l,4),np.uint8)
    square_upper_front_right_arm_img = np.zeros((h_l,w_l,4),np.uint8)
    square_upper_back_right_arm_img = np.zeros((h_l,w_l,4),np.uint8)
    for y in range(h_l):
        for x in range(w_l):
            if y>=sy and y<h+sy and x>=sx and x<w+sx:
                square_upper_right_arm_img[y][x] = upper_right_arm_img[y - sy][x - sx]
                square_upper_front_right_arm_img[y][x] = upper_front_right_arm_img[y - sy][x - sx]
                square_upper_back_right_arm_img[y][x] = upper_back_right_arm_img[y - sy][x - sx]
    
    #２ その服(上腕)画像を上腕なす角だけ回転させる
    #回転の中心を指定  
    center = (int(w_l/2), int(h_l/2))
    #スケールを指定
    scale = 1.0
    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, seihu*right_arm_upper_angle , scale)
    #アフィン変換
    square_upper_right_arm_img = cv2.warpAffine(square_upper_right_arm_img, trans, (w_l,h_l))
    square_upper_front_right_arm_img = cv2.warpAffine(square_upper_front_right_arm_img, trans, (w_l,h_l))
    square_upper_back_right_arm_img = cv2.warpAffine(square_upper_back_right_arm_img, trans, (w_l,h_l))

    '''plt.title("afin")
    plt.imshow(square_upper_right_arm_img)
    plt.show()
    plt.title("afin_front")
    plt.imshow(square_upper_front_right_arm_img)
    plt.show()
    plt.title("afin_back")
    plt.imshow(square_upper_back_right_arm_img)
    plt.show()
'''
    #３ 回転させた服(体)画像をバウンディングボックスサイズで切り取る
    arm_mask = np.zeros((h_l,w_l,1),np.uint8)
    max_y = 0#服画像の服領域の最大y
    min_y = h_l
    max_x = 0
    min_x = w_l
    for y in range(h_l):
        for x in range(w_l):
            if square_upper_right_arm_img[y][x][3] != 0:
                arm_mask[y][x] = 255
                if max_y < y:
                    max_y = y
                if min_y > y:
                    min_y = y
                if max_x < x:
                    max_x = x
                if min_x > x:
                    min_x = x
    print("min_y="+str(min_y))
    print("max_y="+str(max_y))
    print("min_x="+str(min_x))
    print("max_x="+str(max_x))
    print("y="+str(max_y-min_y))
    print("x="+str(max_x-min_x))
    '''plt.title("arm_mask")
    plt.imshow(arm_mask[min_y:max_y,min_x:max_x])
    plt.show()'''

    #４ 人物部位セグム画像を胴体部分のバウンディングボックスサイズを取得する
    human_segm_h,human_segm_w = human_segm.shape[:2]
    human_segm_max_y = 0#服画像の服領域の最大y
    human_segm_min_y = human_segm_h
    human_segm_max_x = 0
    human_segm_min_x = human_segm_w
    for y in range(human_segm_h):
        for x in range(human_segm_w):
            if human_segm[y][x] == part_value:
                if human_segm_max_y < y:
                    human_segm_max_y = y+1
                if human_segm_min_y > y:
                    human_segm_min_y = y
                if human_segm_max_x < x:
                    human_segm_max_x = x+1
                if human_segm_min_x > x:
                    human_segm_min_x = x
    print("human_segm_min_y="+str(human_segm_min_y))
    #print("human_segm_h="+str(human_segm_h))
    print("human_segm_max_y="+str(human_segm_max_y))
    print("human_segm_min_x="+str(human_segm_min_x))
    print("human_segm_max_x="+str(human_segm_max_x))
    print("y="+str(human_segm_max_y-human_segm_min_y))
    print("x="+str(human_segm_max_x-human_segm_min_x))
    print(human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x].shape)
    '''plt.title("human_segm")
    plt.imshow(human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x])
    plt.show()'''
    human_arm = human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#人体部位セグメンテーションの上腕部分
    #result_img = actual_img.copy()#actual_imgをコピーする。これに服を着せる
    #print("result_img.shape="+str(result_img.shape))
    actual_arm = result_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の上腕部分
    human_arm_y,human_arm_x = human_arm.shape[:2]


    #５ 服の体画像を(人物部位セグム画像の胴体の高さ, 服の襟座標を人物部位セグム画像の襟座標に合わさるような横幅)にリサイズする
    resize_y = human_segm_max_y - human_segm_min_y
    resize_x = human_segm_max_x - human_segm_min_x
    arm_img = square_upper_right_arm_img[min_y:max_y,min_x:max_x].copy()
    arm_img = cv2.resize(arm_img, dsize=(resize_x, resize_y), interpolation = cv2.INTER_AREA)#resizeした服の画像でこれを着せる
    front_arm_img = square_upper_front_right_arm_img[min_y:max_y,min_x:max_x].copy()
    front_arm_img = cv2.resize(front_arm_img, dsize=(resize_x, resize_y), interpolation = cv2.INTER_AREA)
    back_arm_img = square_upper_back_right_arm_img[min_y:max_y,min_x:max_x].copy()
    back_arm_img = cv2.resize(back_arm_img, dsize=(resize_x, resize_y), interpolation = cv2.INTER_AREA)
    #print(arm_img.shape)
    '''plt.title("resize_arm_img")
    plt.imshow(arm_img)
    plt.show()
    plt.title("front_arm_img")
    plt.imshow(front_arm_img)
    plt.show()
    plt.title("back_arm_img")
    plt.imshow(back_arm_img)
    plt.show()'''
    
    clothes_min_x = resize_x
    clothes_max_x = 0
    human_min_x = human_segm_max_x
    human_max_x = 0
    for y in range(resize_y):
        clothes_min_x = resize_x#リセットする
        clothes_max_x = 0
        human_min_x = human_segm_max_x
        human_max_x = 0
        print(str(y)+"回")
        for x in range(resize_x):
            if arm_img[y][x][3] != 0:
                if clothes_max_x < x:
                    clothes_max_x = x+1
                if clothes_min_x > x:
                    clothes_min_x = x
        print("clothes_max_x="+str(clothes_max_x))
        print("clothes_min_x="+str(clothes_min_x))
        for foolx in range(human_arm_x):
            if human_segm[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x][y][foolx] == part_value:
                if human_max_x < foolx:
                    human_max_x = foolx+1
                if human_min_x > foolx:
                    human_min_x = foolx
        print("human_max_x="+str(human_max_x))
        print("human_min_x="+str(human_min_x))
        brank = front_arm_img[y:y+1,clothes_min_x:clothes_max_x]
        print(brank)

        brank = cv2.resize(brank,dsize=((human_max_x-human_min_x), 1))#ここで人物部位セグム画像の胴体xサイズにリサイズ
        print(brank)

        print(actual_arm.shape)
        for i in range(human_max_x-human_min_x):#ここで人物部位セグム画像の胴体xサイズにリサイズした画像をresult_imgに貼り付け
            if brank[0][i][3] != 0:
                actual_arm[y][human_min_x+i] = brank[0][i]
    '''plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()'''

    return(result_img)


def right_arm_change(actual_img,result_img,human_segm,candidate,model_name,json_data):
    #右腕
    right_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "right_arm_"+model_name+".png",-1)
    front_right_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "front_right_arm_"+model_name+".png",-1)
    back_right_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "back_right_arm_"+model_name+".png",-1)

    right_shoulder_point = json_data["right_shoulder"] #全体服画像での右肩座標
    right_elbow_point = json_data["right_elbow"]
    right_wrist_point = json_data["right_wrist"]
    right_arm_img_elbow = json_data["right_arm"]["right_elbow"] #服(腕)の右肘座標

    right_arm_upper_vec = np.array(right_shoulder_point) - np.array(right_elbow_point)
    right_arm_lower_vec = np.array(right_wrist_point) - np.array(right_elbow_point)

    a = create_bisector(right_arm_upper_vec,right_arm_lower_vec,right_shoulder_point,right_elbow_point,right_wrist_point)
    
    upper_right_arm_img,lower_right_arm_img,upper_front_right_arm_img,lower_front_right_arm_img,upper_back_right_arm_img,lower_back_right_arm_img = upper_lower_split(right_arm_img,front_right_arm_img,back_right_arm_img,a,right_arm_img_elbow)

    # 人物の骨格情報から腕のベクトルを算出
    actual_right_arm_upper_vec = np.array([int(candidate[2][0]),int(candidate[2][1])]) - np.array([int(candidate[3][0]),int(candidate[3][1])])
    actual_right_arm_lower_vec = np.array([int(candidate[4][0]),int(candidate[4][1])]) - np.array([int(candidate[3][0]),int(candidate[3][1])])
    
    # 服(右上腕)の骨格と人物の骨格との内積から角度を算出
    right_arm_upper_angle = get_angleFrom2Vec(right_arm_upper_vec,actual_right_arm_upper_vec)
    seihu = calc_rotation_direction(right_arm_upper_vec,actual_right_arm_upper_vec,right_arm_upper_angle)
    
    # 服(右上腕)を実際の人物画像に貼り付ける
    part_value = 3
    result_img = adjust_upper_arm_rotate(actual_img,result_img,human_segm,upper_right_arm_img,upper_front_right_arm_img,upper_back_right_arm_img,right_arm_upper_angle,seihu,part_value)

    # 服(右下腕)の骨格と人物の骨格との内積から角度を算出
    right_arm_lower_angle = get_angleFrom2Vec(right_arm_lower_vec,actual_right_arm_lower_vec)
    seihu = calc_rotation_direction(right_arm_lower_vec,actual_right_arm_lower_vec,right_arm_lower_angle)

    # 服(右下腕)を実際の人物画像に貼り付ける
    part_value = 5
    result_img = adjust_upper_arm_rotate(actual_img,result_img,human_segm,lower_right_arm_img,lower_front_right_arm_img,lower_back_right_arm_img,right_arm_lower_angle,seihu,part_value)

    #左腕
    model0 = cv2.imread(CLOTHES_DIR+model_name+"/" +model_name+".png",-1)
    left_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "left_arm_"+model_name+".png",-1)
    front_left_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "front_left_arm_"+model_name+".png",-1)
    back_left_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "back_left_arm_"+model_name+".png",-1)

    left_shoulder_point = json_data["left_shoulder"] #全体服画像での左肩座標
    left_elbow_point = json_data["left_elbow"]
    left_wrist_point = json_data["left_wrist"]
    left_arm_img_elbow = json_data["left_arm"]["left_elbow"] #服(腕)の左肘座標

    left_arm_upper_vec = np.array(left_shoulder_point) - np.array(left_elbow_point)
    left_arm_lower_vec = np.array(left_wrist_point) - np.array(left_elbow_point)

    a = create_bisector(left_arm_upper_vec,left_arm_lower_vec,left_shoulder_point,left_elbow_point,left_wrist_point)
    
    upper_left_arm_img,lower_left_arm_img,upper_front_left_arm_img,lower_front_left_arm_img,upper_back_left_arm_img,lower_back_left_arm_img = upper_lower_split(left_arm_img,front_left_arm_img,back_left_arm_img,a,left_arm_img_elbow)

    # 人物の骨格情報から腕のベクトルを算出
    actual_left_arm_upper_vec = np.array([int(candidate[5][0]),int(candidate[5][1])]) - np.array([int(candidate[6][0]),int(candidate[6][1])])
    actual_left_arm_lower_vec = np.array([int(candidate[7][0]),int(candidate[7][1])]) - np.array([int(candidate[6][0]),int(candidate[6][1])])

    # 服(左上腕)の骨格と人物の骨格との内積から角度を算出
    left_arm_upper_angle = get_angleFrom2Vec(left_arm_upper_vec,actual_left_arm_upper_vec)
    seihu = calc_rotation_direction(left_arm_upper_vec,actual_left_arm_upper_vec,left_arm_upper_angle)

    # 服(左上腕)を実際の人物画像に貼り付ける
    part_value = 2
    result_img = adjust_upper_arm_rotate(actual_img,result_img,human_segm,upper_left_arm_img,upper_front_left_arm_img,upper_back_left_arm_img,left_arm_upper_angle,seihu,part_value)

    # 服(左下腕)の骨格と人物の骨格との内積から角度を算出
    left_arm_lower_angle = get_angleFrom2Vec(left_arm_lower_vec,actual_left_arm_lower_vec)
    seihu = calc_rotation_direction(left_arm_lower_vec,actual_left_arm_lower_vec,left_arm_lower_angle)

    # 服(左下腕)を実際の人物画像に貼り付ける
    part_value = 4
    result_img = adjust_upper_arm_rotate(actual_img,result_img,human_segm,lower_left_arm_img,lower_front_left_arm_img,lower_back_left_arm_img,left_arm_lower_angle,seihu,part_value)

    return(result_img)

def left_arm_change():
    pass

def change(actual_img,human_segm,candidate,model_name):
    # 服の骨格情報などを含むjson_dataをロードする
    with open(CLOTHES_DIR+model_name+"/" + model_name+".json") as f:
        json_data = json.load(f)
    # 着せ替えスタート
    result_img = actual_img.copy()

    result_img = right_arm_change(actual_img,result_img,human_segm,candidate,model_name,json_data)

    result_img = torso_change(actual_img,result_img,human_segm,candidate,model_name,json_data)

    plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()

    return(result_img)