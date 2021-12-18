import os
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt
from src import util
from src.body import Body
 
INPUT_FILE_NAME = "actual_img1.png"

def scale_to_height(img,height):
    """高さが指定した値になるように、アスペクト比を固定して、リサイズする。
    """
    h, w = img.shape[:2]
    width = round(w * (height / h))
    dst = cv2.resize(img, dsize=(width, height))


    return dst
def create_mask(alpha_img):
    mask = np.zeros(alpha_img.shape[:2],dtype=np.uint8)
    h,w = alpha_img.shape[:2]
    for y in range(h):
        for x in range(w):
            if alpha_img[y][x][3] == 255:
                mask[y][x] = 255
    burank = np.zeros(alpha_img.shape[:2],dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    print(contours)
    cv2.drawContours(burank, contours, -1, color=255, thickness=2)
    plt.imshow(burank,cmap="gray",vmin=0,vmax=255)
    plt.show()
    gray = cv2.cvtColor(alpha_img,cv2.COLOR_BGRA2GRAY)
    dst = cv2.Laplacian(gray,cv2.CV_32F, ksize=3)
    plt.imshow(dst,cmap="gray",vmin=0,vmax=255)
    plt.show()
    return burank,dst

def segmentation_arm(img,candidata,burank,lap):
    left_shoulder = 2
    left_elbow = 3
    left_wrist = 4
    left_knee = 8
    right_sholder = 5
    right_elbow = 6
    right_wrist = 7

    left_shoulder_y = int(candidate[left_shoulder][1])
    while True:
        if burank[left_shoulder_y][int(candidate[left_shoulder][0])] == 255 and burank[left_shoulder_y-1][int(candidate[left_shoulder][0])] == 0:
            break
        left_shoulder_y -= 1

    print("candidate[left_shoulder][1]="+str(candidate[left_shoulder][1]))
    print("left_shoulder_y="+str(left_shoulder_y))


    left_x_max = candidata[left_shoulder][0]
    left_x_min = candidata[left_shoulder][0]
    left_y_max = candidata[left_shoulder][1]
    left_y_min = left_shoulder_y
    for i in [left_shoulder,left_elbow,left_wrist,left_knee]:
        print(i)
        if left_x_max < candidata[i][0]:
            left_x_max = candidata[i][0]
        elif left_x_min > candidata[i][0]:
            left_x_min = candidata[i][0]
        if left_y_max < candidata[i][1]:
            left_y_max = candidata[i][1]
        elif left_y_min > candidata[i][1]:
            left_y_min = candidata[i][1]
    
    left_x_max = int(left_x_max)
    left_x_min = int(left_x_min)
    left_y_max = int(left_y_max)
    left_y_min = int(left_y_min)
    print("left_x_max="+str(left_x_max)+",left_x_min="+str(left_x_min)+",left_y_max="+str(left_y_max)+",left_y_min="+str(left_y_min))
    left_arm = img[left_y_min:left_y_max,left_x_min:left_x_max]
    lap_left_arm = lap[left_y_min:left_y_max,left_x_min:left_x_max]
    burank_left_arm = burank[left_y_min:left_y_max,left_x_min:left_x_max]
    plt.imshow(left_arm)
    plt.show()
    plt.imshow(lap_left_arm,cmap="gray",vmin=0,vmax=255)
    plt.show()
    lap_h,lap_w = lap_left_arm.shape[:2]
    for y in range(lap_h):
        for x in range(lap_w):
            if lap_left_arm[y][x] > 120:
                burank_left_arm[y][x] = lap_left_arm[y][x]
    plt.imshow(burank_left_arm,cmap="gray",vmin=0,vmax=255)
    plt.show()                

if __name__ == "__main__":

    body_estimation = Body('model/body_pose_model.pth')
    target_image_path = 'images/' + INPUT_FILE_NAME
    alpha_img = cv2.imread(target_image_path,-1)  # B,G,R,A order

    '''alpha_img = scale_to_height(alpha_img, 500)
    cv2.imwrite("size_500.png",alpha_img)'''
    burank,lap = create_mask(alpha_img)
    oriImg = cv2.cvtColor(alpha_img,cv2.COLOR_BGRA2BGR)

    #oriImg = scale_to_height(oriImg, 500)

    yohaku = 60
    h,w = oriImg.shape[:2]
    waku = np.full((h,w+(yohaku*2),3),255,np.uint8)
    waku_h,waku_w = waku.shape[:2]
    for y in range(waku_h):
        for x in range(waku_w):
            if x >= yohaku and x < w + yohaku:
                waku[y][x] = oriImg[y][x-yohaku]
    

    candidate, subset = body_estimation(waku)
    plt.imshow(waku[0:h,yohaku:waku_w-yohaku])
    plt.show()
    waku = waku[0:h,yohaku:waku_w-yohaku]#余白を削る
    plt.imshow(cv2.cvtColor(waku,cv2.COLOR_BGR2RGB))
    plt.show()

    for i in range(18):
        candidate[i][0] -= yohaku
    print("candidate")
    print(candidate)
    print("subset")
    print(subset)

    segmentation_arm(waku,candidate,burank,lap)

    canvas = copy.deepcopy(waku)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    basename_name = os.path.splitext(os.path.basename(target_image_path))[0]

    result_image_path = "result/pose_" + basename_name + ".jpg"
    cv2.imwrite(result_image_path, canvas)
    plt.imshow(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    plt.show()

