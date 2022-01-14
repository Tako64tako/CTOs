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
    c = 1.0 #角度
    if angle+c > 180: #もしなす角+cが180を超えるようなら
        c = (180 - angle)/2 #cをもっと小さい値にする
    
    # このcの角度だけclothes_vecを正転させたcom_cl_vecとhuman_vecとのなす角度を求める
    sinsin = math.sin(math.radians(c))
    coscos = math.cos(math.radians(c))
    rot_x = (clothes_vec[0] * sinsin) - (clothes_vec[1] * sinsin)
    rot_y = (clothes_vec[0] * sinsin) + (clothes_vec[1] * coscos)
    com_cl_vec = np.array([rot_x,rot_y])
    com_angle = get_angleFrom2Vec(com_cl_vec,human_vec)
    # 正転か後転かのもとになる角度の正負を決める
    check_back = 1 #正転するため角度の正負はプラス
    if com_angle > angle: #もしangleよりcom_angleの方が大きい場合、つまり正転するとなす角は大きくなった場合、なす角でベクトル同士を平行にするには好転する必要がある
        check_back = -1 #後転にするため角度の正負はマイナス
    
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
def adjust_torso_rotate(brank_img,human_segm,torso_img,neck_point_mask,shoulder_formed_angle,json_data,neck_max_width_point,neck_min_width_point):
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
                    max_y = y+1
                if min_y > y:
                    min_y = y
                if max_x < x:
                    max_x = x+1
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
    #actual_torso = result_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の胴体部分
    brank_torso = brank_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の胴体範囲のbrank_img
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

        #print(actual_torso.shape)
        print(brank_torso.shape)
        for i in range(human_max_x-human_min_x):#ここで人物部位セグム画像の胴体xサイズにリサイズした画像をbrank_imgに貼り付け
            if brank[0][i][3] != 0:
                #actual_torso[y][human_min_x+i] = brank[0][i]
                brank_torso[y][human_min_x+i] = brank[0][i]
    '''plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()'''

    return(brank_img)


def torso_change(brank_img,human_segm,candidate,model_name,json_data):
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

    brank_img = adjust_torso_rotate(brank_img,human_segm,torso_img,neck_point_mask,shoulder_formed_angle,json_data,neck_max_width_point,neck_min_width_point)
    return(brank_img)

# 一方のベクトルを、二つのベクトルとのなす角だけ回転させるとき、正転、後転どちらの回転をすれば、
# 他方のベクトルと平行(なす角が0に近く)になるかを調べ、平行になる回転方向を示すフラグを返す。
def findDirectionOfRotation(arm_upper_vec,arm_lower_vec):
    vec_angle = get_angleFrom2Vec(arm_upper_vec,arm_lower_vec)#二つのベクトルのなす角を求める

    # なす角で後転したときのベクトルを求める
    sinsin = math.sin(math.radians(-vec_angle))
    coscos = math.cos(math.radians(-vec_angle))
    backward_rot_x = (arm_upper_vec[0] * sinsin) - (arm_upper_vec[1] * sinsin)
    backward_rot_y = (arm_upper_vec[0] * sinsin) + (arm_upper_vec[1] * coscos)
    backward_line_vec = np.array([backward_rot_x,backward_rot_y])

    # なす角で正転したときのベクトルを求める
    sinsin = math.sin(math.radians(vec_angle))
    coscos = math.cos(math.radians(vec_angle))
    forward_rot_x = (arm_upper_vec[0] * sinsin) - (arm_upper_vec[1] * sinsin)
    forward_rot_y = (arm_upper_vec[0] * sinsin) + (arm_upper_vec[1] * coscos)
    forward_line_vec = np.array([forward_rot_x,forward_rot_y])

    backward_angle = get_angleFrom2Vec(backward_line_vec,arm_lower_vec)# 後転でできたベクトルと他方のベクトルとのなす角を求める
    forward_angle = get_angleFrom2Vec(forward_line_vec,arm_lower_vec)# 正転でできたベクトルと他方のベクトルとのなす角を求める

    print("backward_angle="+str(backward_angle))
    print("forward_angle="+str(forward_angle))
    back_flag = False #回転方向を正転にするフラグ
    #もし後転に回転させたときとのなす角が、正転二回転させたときとのなす角より小さいとき(0に近いほうが回転方向)
    if backward_angle < forward_angle:
        back_flag = True #回転方向を後転にするフラグ
    return back_flag,vec_angle

# 全体服画像の腕部分をlowerとupperを二等分する線の式を作成する
def create_bisector(arm_upper_vec,arm_lower_vec,shoulder_point,elbow_point,wrist_point):
    back_flag,vec_angle = findDirectionOfRotation(arm_upper_vec,arm_lower_vec)# 正転、後転のどちらの回転をすれば良いか調べる
    arm_angle = vec_angle / 2
    print(arm_angle)
    if back_flag==True:
        sinsin = math.sin(math.radians(-arm_angle))
        coscos = math.cos(math.radians(-arm_angle))
        rot_x = (arm_upper_vec[0] * sinsin) - (arm_upper_vec[1] * sinsin)
        rot_y = (arm_upper_vec[0] * sinsin) + (arm_upper_vec[1] * coscos)
    else:
        sinsin = math.sin(math.radians(arm_angle))
        coscos = math.cos(math.radians(arm_angle))
        rot_x = (arm_upper_vec[0] * sinsin) - (arm_upper_vec[1] * sinsin)
        rot_y = (arm_upper_vec[0] * sinsin) + (arm_upper_vec[1] * coscos)
    
    line_point = np.array([rot_x,rot_y]) #+ np.array(right_elbow_point)
    print("line_point="+str(line_point))
    a = (-line_point[1])/(-line_point[0]) # a = (Px1-Px2)/(Py1-Py2)
    #b = 0 - (a * 0)
    print("a="+str(a))
    #print("b="+str(b))
    return(a)

# 服(腕)画像、front服(腕)画像、back服(腕)画像をlower,upperを分けた画像をそれぞれ生成する
def upper_lower_split(arm_img,front_arm_img,a,arm_img_elbow):
    '''plt.title("arm_img")
    plt.imshow(arm_img)
    plt.show()'''
    h,w = arm_img.shape[:2]
    upper_arm_img = np.zeros((h,w,4),np.uint8)
    lower_arm_img = np.zeros((h,w,4),np.uint8)
    upper_front_arm_img = np.zeros((h,w,4),np.uint8)
    lower_front_arm_img = np.zeros((h,w,4),np.uint8)
    #upper_back_arm_img = np.zeros((h,w,4),np.uint8)
    #lower_back_arm_img = np.zeros((h,w,4),np.uint8)
    
    for y in range(h):
        for x in range(w):
            if arm_img[y][x][3] == 255:
                if (a*(x-arm_img_elbow[0])+arm_img_elbow[1])<y:
                    lower_arm_img[y][x][3] = 255
                else:
                    upper_arm_img[y][x][3] = 255


            if front_arm_img[y][x][3] == 255:
                if (a*(x-arm_img_elbow[0])+arm_img_elbow[1])<y:
                    lower_front_arm_img[y][x][3] = 255
                else:
                    upper_front_arm_img[y][x][3] = 255

            '''if back_arm_img[y][x][3] == 255:
                if (a*(x-arm_img_elbow[0])+arm_img_elbow[1])<y:
                    lower_back_arm_img[y][x][3] = 255
                else:
                    upper_back_arm_img[y][x][3] = 255'''
    
    for y in range(h):
        for x in range(w):
            if arm_img[y][x][3] == 255 and upper_arm_img[y][x][3] == 255:
                upper_arm_img[y][x] = arm_img[y][x]
            elif arm_img[y][x][3] == 255 and lower_arm_img[y][x][3] == 255:
                lower_arm_img[y][x] = arm_img[y][x]
            
            if front_arm_img[y][x][3] == 255 and upper_front_arm_img[y][x][3] == 255:
                upper_front_arm_img[y][x] = front_arm_img[y][x]
            elif front_arm_img[y][x][3] == 255 and lower_front_arm_img[y][x][3] == 255:
                lower_front_arm_img[y][x] = front_arm_img[y][x]

            '''if back_arm_img[y][x][3] == 255 and upper_back_arm_img[y][x][3] == 255:
                upper_back_arm_img[y][x] = back_arm_img[y][x]
            elif back_arm_img[y][x][3] == 255 and lower_back_arm_img[y][x][3] == 255:
                lower_back_arm_img[y][x] = back_arm_img[y][x]'''
            

    '''plt.title("upper_arm_img")
    plt.imshow(upper_arm_img)
    plt.show()
    plt.title("lower_arm_img")
    plt.imshow(lower_arm_img)
    plt.show()
    plt.title("upper_front_arm_img")
    plt.imshow(upper_front_arm_img)
    plt.show()
    plt.title("lower_front_arm_img")
    plt.imshow(lower_front_arm_img)
    plt.show()
    plt.title("upper_back_arm_img")
    plt.imshow(upper_back_arm_img)
    plt.show()
    plt.title("lower_back_arm_img")
    plt.imshow(lower_back_arm_img)
    plt.show()'''

    return(upper_arm_img,lower_arm_img,upper_front_arm_img,lower_front_arm_img)

def adjust_arm_rotate(brank_img,human_segm,upper_arm_img,upper_front_arm_img,arm_angle,seihu,part_value):
    #１ 服(上腕)画像の横縦幅をh+wの正方形にした画像を作成する※回転した際に服(上腕)自体が画像からはみ出ないようにするため
    h,w = upper_arm_img.shape[:2]
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
    square_upper_arm_img = np.zeros((h_l,w_l,4),np.uint8)
    square_upper_front_arm_img = np.zeros((h_l,w_l,4),np.uint8)
    #square_upper_back_arm_img = np.zeros((h_l,w_l,4),np.uint8)
    for y in range(h_l):
        for x in range(w_l):
            if y>=sy and y<h+sy and x>=sx and x<w+sx:
                square_upper_arm_img[y][x] = upper_arm_img[y - sy][x - sx]
                square_upper_front_arm_img[y][x] = upper_front_arm_img[y - sy][x - sx]
                #square_upper_back_arm_img[y][x] = upper_back_arm_img[y - sy][x - sx]
    
    #２ その服(上腕)画像を上腕なす角だけ回転させる
    #回転の中心を指定  
    center = (int(w_l/2), int(h_l/2))
    #スケールを指定
    scale = 1.0
    #getRotationMatrix2D関数を使用
    trans = cv2.getRotationMatrix2D(center, seihu*arm_angle , scale)
    #アフィン変換
    square_upper_arm_img = cv2.warpAffine(square_upper_arm_img, trans, (w_l,h_l))
    square_upper_front_arm_img = cv2.warpAffine(square_upper_front_arm_img, trans, (w_l,h_l))
    #square_upper_back_arm_img = cv2.warpAffine(square_upper_back_arm_img, trans, (w_l,h_l))

    '''plt.title("afin")
    plt.imshow(square_upper_arm_img)
    plt.show()
    plt.title("afin_front")
    plt.imshow(square_upper_front_arm_img)
    plt.show()
    plt.title("afin_back")
    plt.imshow(square_upper_back_arm_img)
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
            if square_upper_arm_img[y][x][3] != 0:
                arm_mask[y][x] = 255
                if max_y < y:
                    max_y = y+1#この+1は[min_y:max_y]をするため
                if min_y > y:
                    min_y = y
                if max_x < x:
                    max_x = x+1#この+1は[min_x:max_x]をするため
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
    #actual_arm = result_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の上腕部分
    brank_arm = brank_img[human_segm_min_y:human_segm_max_y,human_segm_min_x:human_segm_max_x]#実際の人物画像の上腕領域にあたるbrank_img
    human_arm_y,human_arm_x = human_arm.shape[:2]


    #５ 服の体画像を(人物部位セグム画像の胴体の高さ, 服の襟座標を人物部位セグム画像の襟座標に合わさるような横幅)にリサイズする
    resize_y = human_segm_max_y - human_segm_min_y
    resize_x = human_segm_max_x - human_segm_min_x
    arm_img = square_upper_arm_img[min_y:max_y,min_x:max_x].copy()
    arm_img = cv2.resize(arm_img, dsize=(resize_x, resize_y), interpolation = cv2.INTER_AREA)#resizeした服の画像でこれを着せる
    front_arm_img = square_upper_front_arm_img[min_y:max_y,min_x:max_x].copy()
    front_arm_img = cv2.resize(front_arm_img, dsize=(resize_x, resize_y), interpolation = cv2.INTER_AREA)
    #back_arm_img = square_upper_back_arm_img[min_y:max_y,min_x:max_x].copy()
    #back_arm_img = cv2.resize(back_arm_img, dsize=(resize_x, resize_y), interpolation = cv2.INTER_AREA)
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

        #print(actual_arm.shape)
        print(brank_arm.shape)
        for i in range(human_max_x-human_min_x):#ここで人物部位セグム画像の胴体xサイズにリサイズした画像をbrank_imgに貼り付け
            if brank[0][i][3] != 0:
                #actual_arm[y][human_min_x+i] = brank[0][i]
                brank_arm[y][human_min_x+i] = brank[0][i]
    '''plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()'''

    return(brank_img)


def arm_change(brank_img,human_segm,candidate,model_name,json_data):
    #右腕
    right_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "right_arm_"+model_name+".png",-1)
    front_right_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "front_right_arm_"+model_name+".png",-1)
    #back_right_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "back_right_arm_"+model_name+".png",-1)

    right_shoulder_point = json_data["right_shoulder"] #全体服画像での右肩座標
    right_elbow_point = json_data["right_elbow"]
    right_wrist_point = json_data["right_wrist"]
    right_arm_img_elbow = json_data["right_arm"]["right_elbow"] #服(腕)の右肘座標

    right_arm_upper_vec = np.array(right_shoulder_point) - np.array(right_elbow_point)
    right_arm_lower_vec = np.array(right_wrist_point) - np.array(right_elbow_point)

    a = create_bisector(right_arm_upper_vec,right_arm_lower_vec,right_shoulder_point,right_elbow_point,right_wrist_point)
    
    upper_right_arm_img,lower_right_arm_img,upper_front_right_arm_img,lower_front_right_arm_img = upper_lower_split(right_arm_img,front_right_arm_img,a,right_arm_img_elbow)

    # 人物の骨格情報から腕のベクトルを算出
    actual_right_arm_upper_vec = np.array([int(candidate[2][0]),int(candidate[2][1])]) - np.array([int(candidate[3][0]),int(candidate[3][1])])
    actual_right_arm_lower_vec = np.array([int(candidate[4][0]),int(candidate[4][1])]) - np.array([int(candidate[3][0]),int(candidate[3][1])])
    
    # 服(右上腕)の骨格と人物の骨格との内積から角度を算出
    right_arm_upper_angle = get_angleFrom2Vec(right_arm_upper_vec,actual_right_arm_upper_vec)
    seihu = calc_rotation_direction(right_arm_upper_vec,actual_right_arm_upper_vec,right_arm_upper_angle)
    
    # 服(右上腕)を実際の人物画像に合わせてbrank画像に貼り付ける
    part_value = 3
    brank_img = adjust_arm_rotate(brank_img,human_segm,upper_right_arm_img,upper_front_right_arm_img,right_arm_upper_angle,seihu,part_value)

    # 服(右下腕)の骨格と人物の骨格との内積から角度を算出
    right_arm_lower_angle = get_angleFrom2Vec(right_arm_lower_vec,actual_right_arm_lower_vec)
    seihu = calc_rotation_direction(right_arm_lower_vec,actual_right_arm_lower_vec,right_arm_lower_angle)

    # 服(右下腕)を実際の人物画像に貼り付ける
    part_value = 5
    brank_img = adjust_arm_rotate(brank_img,human_segm,lower_right_arm_img,lower_front_right_arm_img,right_arm_lower_angle,seihu,part_value)

    #左腕
    model0 = cv2.imread(CLOTHES_DIR+model_name+"/" +model_name+".png",-1)
    left_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "left_arm_"+model_name+".png",-1)
    front_left_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "front_left_arm_"+model_name+".png",-1)
    #back_left_arm_img = cv2.imread(CLOTHES_DIR+model_name+"/" + "back_left_arm_"+model_name+".png",-1)

    left_shoulder_point = json_data["left_shoulder"] #全体服画像での左肩座標
    left_elbow_point = json_data["left_elbow"]
    left_wrist_point = json_data["left_wrist"]
    left_arm_img_elbow = json_data["left_arm"]["left_elbow"] #服(腕)の左肘座標

    left_arm_upper_vec = np.array(left_shoulder_point) - np.array(left_elbow_point)
    left_arm_lower_vec = np.array(left_wrist_point) - np.array(left_elbow_point)

    a = create_bisector(left_arm_upper_vec,left_arm_lower_vec,left_shoulder_point,left_elbow_point,left_wrist_point)
    
    upper_left_arm_img,lower_left_arm_img,upper_front_left_arm_img,lower_front_left_arm_img = upper_lower_split(left_arm_img,front_left_arm_img,a,left_arm_img_elbow)

    # 人物の骨格情報から腕のベクトルを算出
    actual_left_arm_upper_vec = np.array([int(candidate[5][0]),int(candidate[5][1])]) - np.array([int(candidate[6][0]),int(candidate[6][1])])
    actual_left_arm_lower_vec = np.array([int(candidate[7][0]),int(candidate[7][1])]) - np.array([int(candidate[6][0]),int(candidate[6][1])])

    # 服(左上腕)の骨格と人物の骨格との内積から角度を算出
    left_arm_upper_angle = get_angleFrom2Vec(left_arm_upper_vec,actual_left_arm_upper_vec)
    seihu = calc_rotation_direction(left_arm_upper_vec,actual_left_arm_upper_vec,left_arm_upper_angle)

    # 服(左上腕)を実際の人物画像に貼り付ける
    part_value = 2
    brank_img = adjust_arm_rotate(brank_img,human_segm,upper_left_arm_img,upper_front_left_arm_img,left_arm_upper_angle,seihu,part_value)

    # 服(左下腕)の骨格と人物の骨格との内積から角度を算出
    left_arm_lower_angle = get_angleFrom2Vec(left_arm_lower_vec,actual_left_arm_lower_vec)
    seihu = calc_rotation_direction(left_arm_lower_vec,actual_left_arm_lower_vec,left_arm_lower_angle)

    # 服(左下腕)を実際の人物画像に貼り付ける
    part_value = 4
    brank_img = adjust_arm_rotate(brank_img,human_segm,lower_left_arm_img,lower_front_left_arm_img,left_arm_lower_angle,seihu,part_value)

    return(brank_img)

# これはbrank_imgを補正する
def bondingCorrection(brank_img,human_segm,model_name):
    h,w = human_segm.shape[:2]
    boding_img = np.zeros((h,w,1),np.uint8)#これは人物画像と同じサイズの、補正する箇所を示す画像
    kernel = [[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]]#ピクセルから探索する場所リスト
    segm_kernel = [5,3,8,2,4]#これはhuman_segmにおける補正する部位を示すリスト
    # まず、human_segmのsegm_kernelに属するピクセルのうち、部位と部位のつなぎ目になるピクセルを探し、boding_imgの同ピクセルに部位値*部位値を入れる
    for y in range(h):
        for x in range(w):
            if human_segm[y][x] in segm_kernel:
                for point in kernel:
                    karas_y = y+point[1]
                    karas_x = x+point[0]
                    if karas_y>=0 and karas_y<h and karas_x>=0 and karas_x<w:#out of index error対策
                        if human_segm[karas_y][karas_x] != human_segm[y][x] and human_segm[karas_y][karas_x] in segm_kernel:
                            boding_img[y][x] = human_segm[y][x] * human_segm[karas_y][karas_x]
                            break

    ter_kernel = [15,24,16,8]#boding_imgに存在するピクセル値のリスト
    point_kernel = []#boding_imgの各ter_kernelに属するピクセル値から取得した短形領域になる[min_y,max_y,min_x,max_x]を格納するリスト
    # 各ter_kernelに属するピクセル値から取得した短形領域になる[min_y,max_y,min_x,max_x]をpoint_kernelに格納する
    for ter_n in ter_kernel:
        max_x = 0
        min_x = w
        max_y = 0
        min_y = h
        for y in range(h):
            for x in range(w):
                if boding_img[y][x] == ter_n:
                    if max_x < x:
                        max_x = x+1
                    if min_x > x:
                        min_x = x
                    if max_y < y:
                        max_y = y+1
                    if min_y > y:
                        min_y = y
        if min_x >= 2:
            min_x -= 2
        if max_x <= w-2:
            max_x += 2
        if min_y >= 2:
            min_y -= 2
        if max_y <= h-2:
            max_y += 2
        print(ter_n)
        print([min_y,max_y,min_x,max_x])
        point_kernel.append([min_y,max_y,min_x,max_x])
    
    print(point_kernel)
    # point_kernelの要素から、画像を短形領域の範囲に抽出する。
    # shape_boding_imgがter_kernelに属していないピクセル かつ shape_brank_imgでアルファが100未満のピクセル かつ
    # shape_human_segmでsegm_kernelに属しているピクセルの場合、shape_boding_imgのそのピクセルを255にする
    for shape in point_kernel:
        shape_human_segm = human_segm[shape[0]:shape[1],shape[2]:shape[3]]
        shape_brank_img = brank_img[shape[0]:shape[1],shape[2]:shape[3]]
        shape_boding_img = boding_img[shape[0]:shape[1],shape[2]:shape[3]]
        h,w = shape_human_segm.shape[:2]
        for y in range(h):
            for x in range(w):
                if shape_boding_img[y][x] not in ter_kernel and shape_brank_img[y][x][3]<230 and shape_human_segm[y][x] in segm_kernel:
                    shape_boding_img[y][x] = 255
    
    h,w = boding_img.shape[:2]
    '''for y in range(h):
        for x in range(w):
            if boding_img[y][x] in ter_kernel:
                boding_img[y][x] = 255'''
    brank_img_copy = brank_img.copy()
    for y in range(h):
        for x in range(w):
            if boding_img[y][x] in ter_kernel or boding_img[y][x] == 255:
                kernel_y = -3
                kernel_x = -3
                num = 0
                pixel_sum = np.array([0,0,0,0])
                for a_y in range(7):
                    for a_x in range(7):
                        karas_y = y+kernel_y+a_y
                        karas_x = x+kernel_x+a_x
                        if karas_y>=0 and karas_y<h and karas_x>=0 and karas_x<w:#out of index error対策
                            if human_segm[karas_y][karas_x] in segm_kernel and brank_img[karas_y][karas_x][3] > 230:
                                pixel_sum+=brank_img[karas_y][karas_x]
                                num+=1
                pixel_sum = (pixel_sum/num).astype(np.uint8)
                brank_img[y][x] = pixel_sum
    for y in range(h):
        for x in range(w):
            if boding_img[y][x] in ter_kernel or boding_img[y][x] == 255:
                kernel_y = -3
                kernel_x = -3
                num = 0
                pixel_sum = np.array([0,0,0,0])
                for a_y in range(7):
                    for a_x in range(7):
                        karas_y = y+kernel_y+a_y
                        karas_x = x+kernel_x+a_x
                        if karas_y>=0 and karas_y<h and karas_x>=0 and karas_x<w:#out of index error対策
                            if human_segm[karas_y][karas_x] in segm_kernel and brank_img[karas_y][karas_x][3] > 230:
                                pixel_sum+=brank_img[karas_y][karas_x]
                                num+=1
                pixel_sum = (pixel_sum/num).astype(np.uint8)
                brank_img[y][x] = pixel_sum

    return(brank_img,boding_img)

# human_segmに合わせた服画像であるblank_imgを背景削除人物画像に貼り付ける
def mounting(result_img,brank_img):
    h,w = brank_img.shape[:2]
    for y in range(h):
        for x in range(w):
            if brank_img[y][x][3] != 0:
                result_img[y][x] = brank_img[y][x]
    return(result_img)
    
def change(actual_img,human_segm,candidate,model_name):
    # 服の骨格情報などを含むjson_dataをロードする
    with open(CLOTHES_DIR+model_name+"/" + model_name+".json") as f:
        json_data = json.load(f)
    # 着せ替えスタート
    result_img = actual_img.copy()

    brank_img = np.zeros(result_img.shape,np.uint8)

    brank_img = arm_change(brank_img,human_segm,candidate,model_name,json_data)

    brank_img = torso_change(brank_img,human_segm,candidate,model_name,json_data)

    brank_img,boding_img = bondingCorrection(brank_img,human_segm,model_name)
    #cv2.imwrite("./DressApp/dress_lib/images/temporary_imgs/brank_"+model_name+".png",brank_img)
    #cv2.imwrite("./DressApp/dress_lib/images/temporary_imgs/boding_"+model_name+".png",boding_img)

    result_img = mounting(result_img,brank_img)
    plt.title("result")
    plt.imshow(cv2.cvtColor(result_img,cv2.COLOR_BGRA2RGBA))
    plt.show()
    
    return(result_img,brank_img)