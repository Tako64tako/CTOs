import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from PIL import Image
import cv2
from .utils import load_graph_model, get_input_tensors, get_output_tensors
import tensorflow as tf
# make tensorflow stop spamming messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


# PATHS
#imagePath = './simple_bodypix_python_master/input_img/rs_bler_IMG_0137_risize.png'
outdirPath = "./DressApp/dress_lib/images/part_segm_images/"
modelPath = './DressApp/dress_lib/simple_bodypix_python_master/bodypix_resnet50_float_model-stride16/model.json'

# CONSTANTS
OutputStride = 16

KEYPOINT_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]


KEYPOINT_IDS = {name: id for id, name in enumerate(KEYPOINT_NAMES)}

CONNECTED_KEYPOINTS_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

CONNECTED_KEYPOINT_INDICES = [(KEYPOINT_IDS[a], KEYPOINT_IDS[b])
                              for a, b in CONNECTED_KEYPOINTS_NAMES]

PART_CHANNELS = [
    'left_face',
    'right_face',
    'left_upper_arm_front',
    'left_upper_arm_back',
    'right_upper_arm_front',
    'right_upper_arm_back',
    'left_lower_arm_front',
    'left_lower_arm_back',
    'right_lower_arm_front',
    'right_lower_arm_back',
    'left_hand',
    'right_hand',
    'torso_front',
    'torso_back',
    'left_upper_leg_front',
    'left_upper_leg_back',
    'right_upper_leg_front',
    'right_upper_leg_back',
    'left_lower_leg_front',
    'left_lower_leg_back',
    'right_lower_leg_front',
    'right_lower_leg_back',
    'left_feet',
    'right_feet'
]
NEW_PART_CHANNELS = [
    'face',
    'left_upper_arm',
    'right_upper_arm',
    'left_lower_arm',
    'right_lower_arm',
    'left_hand',
    'right_hand',
    'torso',
    'left_upper_leg',
    'right_upper_leg',
    'left_lower_leg',
    'right_lower_leg',
    'left_feet',
    'right_feet'
]


print("Loading model...", end="")
graph = load_graph_model(modelPath)  # downloaded from the link above
print("done.\nLoading sample image...", end="")

#画像をリサイズする
def humanImgSize_decide(body_height,imgWidth,imgHeight):
    thr = 350
    human_img_height = int((body_height+thr)//OutputStride)*OutputStride + 1
    hi = human_img_height / imgHeight
    human_img_width = int(int(imgWidth * hi)//OutputStride)*OutputStride + 0
    return human_img_width,human_img_height

#人物部位セグメンテーション画像を部位ごとに領域抽出する。領域抽出して領域が二つ以上なら小さい方
# を隣接する部位(最も隣接数が多い部位)で塗りつぶし調整する。
def adjust_human_segm(human_segm):
    for p in range(len(NEW_PART_CHANNELS)):
        #print(NEW_PART_CHANNELS[p])
        #print(p+1)
        h,w = human_segm.shape[:2]
        two_valued_img = np.zeros((h,w,1),np.uint8)
        for y in range(h):
            for x in range(w):
                if human_segm[y][x] == p+1:
                    two_valued_img[y][x] = 255
        contours, hierarchy = cv2.findContours(two_valued_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        my_segm = p+1
        kensaku = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]
        ta_segm = []
        sum_segm = []
        #print(len(contours))
        if len(contours) != 1:
            max_sum_shape = int(contours[0].shape[0])
            max_sum_shape_ind = 0
            for ey in range(len(contours)):
                if max_sum_shape < int(contours[ey].shape[0]):
                    max_sum_shape = int(contours[ey].shape[0])
                    max_sum_shape_ind = ey

            for ey in range(len(contours)):
                ta_segm = []
                sum_segm = []
                if max_sum_shape_ind != ey:
                    #print(contours[ey][0][0][0])
                    #print(type(int(contours[ey][0][0][0])))
                    for ex in range(len(contours[ey])):
                        for n_list in kensaku:
                            #print("contours[ey][ex][0]="+str(contours[ey][ex][0][1]))
                            #print("n_list[1]="+str(n_list[1]))
                            segm_num = human_segm[int(contours[ey][ex][0][1])+n_list[1]][int(contours[ey][ex][0][0])+n_list[0]]
                            if segm_num != 0 and segm_num != my_segm:
                                if segm_num in ta_segm:
                                    indeindeindex =  ta_segm.index(segm_num)
                                    sum_segm[indeindeindex] += 1
                                else:
                                    ta_segm.append(int(segm_num))
                                    sum_segm.append(1)
                    #print(contours[ey].shape)
                    #print("ta_segm="+str(ta_segm))
                    #print("sum_segm="+str(sum_segm))
                    max_human_segm_num = ta_segm[sum_segm.index(max(sum_segm))]
                    #print(np.array([contours[ey]]))
                    cv2.fillPoly(human_segm,np.array([contours[ey]]),max_human_segm_num)
    
    segm_contours_list = []
    for p in range(len(NEW_PART_CHANNELS)):
        h,w = human_segm.shape[:2]
        two_valued_img = np.zeros((h,w,1),np.uint8)
        for y in range(h):
            for x in range(w):
                if human_segm[y][x] == p+1:
                    two_valued_img[y][x] = 255
        contours, hierarchy = cv2.findContours(two_valued_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        segm_contours_list.append(contours[0])
    
    segm_contours_list = np.array(segm_contours_list)
    return human_segm,segm_contours_list

#人物部位セグメンテーション画像に対して、人物画像からはみ出た部分を消す
def adjust_actual_cut(actualimg,human_segm):
    h,w = actualimg.shape[:2]
    for y in range(h):
        for x in range(w):
            if actualimg[y][x][3] == 0 and human_segm[y][x] != 0:
                human_segm[y][x] = 0
    return human_segm

#人物部位セグメンテーション画像に対して、人物画像から部位領域を拡張する
def adjust_actual_add(actualimg,human_segm,segm_contours_list):
    #right_lower_arm 5
    '''max_height = int(segm_contours_list[5-1][0][0][1])
    min_height = int(segm_contours_list[5-1][0][0][1])
    max_width = int(segm_contours_list[5-1][0][0][0])
    min_width = int(segm_contours_list[5-1][0][0][0])
    print("max_height="+str(max_height))
    print("min_height="+str(min_height))
    print("max_width="+str(max_width))
    print("min_width="+str(min_width))
    for contours_point in segm_contours_list[4]:
        if max_height < int(contours_point[0][1]):
            max_height = int(contours_point[0][1])
        elif min_height > int(contours_point[0][1]):
            min_height = int(contours_point[0][1])
        if max_width < int(contours_point[0][0]):
            max_width = int(contours_point[0][0])
        elif min_width > int(contours_point[0][0]):
            min_width = int(contours_point[0][0])'''
    #kensaku = [[-1,0],[-1,-1],[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1]]
    kensaku = [[-1,0],[0,-1],[1,0],[0,1]]
    ta_segm = []#隣接するピクセルの部位番号を格納するリスト
    human_segm_copy = human_segm.copy()#人物セグメンテーション画像を複製
    finish_flag = 0#ループ停止を意味するフラグ
    h,w = human_segm.shape[:2]

    #人物部位セグメンテーション画像に対して、人物画像から部位領域を拡張する
    #変更点が見つかったらまずコピーの方のピクセルを変える
    #一回ループが終わった後、コピーから変更結果を実物の方に適応させる
    #こうしないとセグメンテーションがふしぜんになる。後に処理される部位が不利になる
    while finish_flag == 0:
        finish_flag = 1
        for y in range(h):
            for x in range(w):
                if actualimg[y][x][3] == 255 and human_segm[y][x] == 0:
                    ta_segm = []#初期化
                    for n_list in kensaku:
                        karas_y = y+n_list[1]
                        karas_x = x+n_list[0]
                        if karas_y>=0 and karas_y<h and karas_x>=0 and karas_x<w:#out of index error対策
                            if human_segm[karas_y][karas_x] != 0:
                                if not(human_segm[karas_y][karas_x] in ta_segm):
                                    ta_segm.append(human_segm[karas_y][karas_x])
                            if len(ta_segm) == 1:
                                human_segm_copy[y][x] = ta_segm[0]
                                finish_flag = 0
        #human_segm = human_segm_copy.copy()
        #変更の適応処理
        for y in range(h):
            for x in range(w):
                if human_segm[y][x] != human_segm_copy[y][x]:
                    human_segm[y][x] = human_segm_copy[y][x]
    return human_segm

#人物セグメンテーション画像をクロージングしてノイズを消す
def dilated_human_segm(human_segm,k_size=(5,5),ite=1):
    order_dilated = [7,6,14,13,5,4,3,2,12,11,10,9,1,8]
    h,w = human_segm.shape[:2]
    adjust_img = np.zeros((h,w,1),np.uint8)
    for i in order_dilated:
        thr_img = np.zeros((h,w,1),np.uint8)
        thr_img[human_segm == i] = 255
        '''plt.title('human_segm_dilated')
        plt.imshow(thr_img,cmap='gray')
        plt.show()'''
        kernel = np.ones(k_size,np.uint8) #要素が全て1の配列を生成
        thr_img = cv2.dilate(thr_img,kernel,iterations = ite)
        thr_img = cv2.erode(thr_img,kernel,iterations = ite)
        adjust_img[thr_img == 255] = i
    return adjust_img


def getBoundingBox(keypointPositions, offset=(10, 10, 10, 10)):
    minX = math.inf
    minY = math.inf
    maxX = - math.inf
    maxY = -math.inf
    for x, y in keypointPositions:
        if (x < minX):
            minX = x
        if(y < minY):
            minY = y
        if(x > maxX):
            maxX = x
        if (y > maxY):
            maxY = y
    return (minX - offset[0], minY-offset[1]), (maxX+offset[2], maxY + offset[3])

def segm_run(alpha_img,height):
    # load sample image into numpy array
    #imagePath = alpha_img_path
    #img = tf.keras.preprocessing.image.load_img(imagePath)
    img = cv2.cvtColor(alpha_img,cv2.COLOR_BGRA2BGR)
    img = tf.keras.preprocessing.image.array_to_img(img, scale=True)
    '''with open("./DressApp/log.txt", mode='a') as f:# python側の処理が見えるようにログファイルに書き込み
        f.write("tf.keras.preprocessing.image.load_img(imagePath)")
        f.write("\n")
        f.write(str(img.getpixel((0,0))))
        f.write("\n")'''
    imgWidth, imgHeight = img.size
    print("imgWidth"+str(imgWidth))
    print("imgHeight"+str(imgHeight))
    #actualWidth,actualHeight = humanImgSize_decide(165,imgWidth,imgHeight)
    actualWidth,actualHeight = humanImgSize_decide(height,imgWidth,imgHeight)

    targetWidth = (int(imgWidth) // OutputStride) * OutputStride + 1
    targetHeight = (int(imgHeight) // OutputStride) * OutputStride + 1
    print("targetWidth"+str(targetWidth))
    print("targetHeight"+str(targetHeight))

    print(imgHeight, imgWidth, targetHeight, targetWidth)
    img = img.resize((targetWidth, targetHeight))

    actualimg = img.resize((actualWidth, actualHeight))

    x = tf.keras.preprocessing.image.img_to_array(img, dtype=np.float32)
    InputImageShape = x.shape
    print("Input Image Shape in hwc", InputImageShape)


    widthResolution = int((InputImageShape[1] - 1) / OutputStride) + 1
    heightResolution = int((InputImageShape[0] - 1) / OutputStride) + 1
    print('Resolution', widthResolution, heightResolution)

    # Get input and output tensors
    input_tensor_names = get_input_tensors(graph)
    print(input_tensor_names)
    output_tensor_names = get_output_tensors(graph)
    print(output_tensor_names)
    input_tensor = graph.get_tensor_by_name(input_tensor_names[0])
    print("input_tensor.shape="+str(input_tensor))

    # Preprocessing Image
    # For Resnet
    if any('resnet_v1' in name for name in output_tensor_names):
        # add imagenet mean - extracted from body-pix source
        m = np.array([-123.15, -115.90, -103.06])
        x = np.add(x, m)
    # For Mobilenet
    elif any('MobilenetV1' in name for name in output_tensor_names):
        x = (x/127.5)-1
    else:
        print('Unknown Model')
    print("x.shape="+str(x.shape))
    sample_image = x[tf.newaxis, ...]
    print("sample_image.shape="+str(sample_image.shape))

    print("done.\nRunning inference...", end="")

    # evaluate the loaded model directly
    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_tensor_names, feed_dict={
                        input_tensor: sample_image})
    print("done. {} outputs received".format(len(results)))  # should be 8 outputs

    for idx, name in enumerate(output_tensor_names):
        if 'displacement_bwd' in name:
            print('displacement_bwd', results[idx].shape)
        elif 'displacement_fwd' in name:
            print('displacement_fwd', results[idx].shape)
        elif 'float_heatmaps' in name:
            heatmaps = np.squeeze(results[idx], 0)
            print('heatmaps', heatmaps.shape)
        elif 'float_long_offsets' in name:
            longoffsets = np.squeeze(results[idx], 0)
            print('longoffsets', longoffsets.shape)
        elif 'float_short_offsets' in name:
            offsets = np.squeeze(results[idx], 0)
            print('offests', offsets.shape)
        elif 'float_part_heatmaps' in name:
            partHeatmaps = np.squeeze(results[idx], 0)
            print('partHeatmaps', partHeatmaps.shape)
        elif 'float_segments' in name:
            segments = np.squeeze(results[idx], 0)
            print('segments', segments.shape)
        elif 'float_part_offsets' in name:
            partOffsets = np.squeeze(results[idx], 0)
            print('partOffsets', partOffsets.shape)
        else:
            print('Unknown Output Tensor', name, idx)


    # BODYPART SEGMENTATION
    partOffsetVector = []
    partHeatmapPositions = []
    partPositions = []
    partScores = []
    partMasks = []

    # Segmentation MASk
    segmentation_threshold = 0.7
    segmentScores = tf.sigmoid(segments)
    mask = tf.math.greater(segmentScores, tf.constant(segmentation_threshold))
    print('maskshape', mask.shape)
    segmentationMask = tf.dtypes.cast(mask, tf.int32)
    segmentationMask = np.reshape(
        segmentationMask, (segmentationMask.shape[0], segmentationMask.shape[1]))
    print('maskValue', segmentationMask[:][:])

    plt.clf()
    plt.title('Segmentation Mask')
    plt.ylabel('y')
    plt.xlabel('x')
    print("segmentationMask.shape="+str(segmentationMask.shape))
    '''plt.imshow(segmentationMask * OutputStride)
    plt.show()'''

    # actualセグメンテーション
    mask_img = Image.fromarray(segmentationMask * 255)
    mask_img = mask_img.resize(
        (actualWidth, actualHeight), Image.LANCZOS).convert("RGB")
    mask_img = tf.keras.preprocessing.image.img_to_array(
        mask_img, dtype=np.uint8)

    segmentationMask_inv = np.bitwise_not(mask_img)
    fg = np.bitwise_and(np.array(actualimg), np.array(
        mask_img))
    '''plt.title('Actual Foreground Segmentation')
    plt.imshow(fg)
    plt.show()'''
    bg = np.bitwise_and(np.array(actualimg), np.array(
        segmentationMask_inv))
    '''plt.title('Actual Background Segmentation')
    plt.imshow(bg)
    plt.show()
    '''
    # セグメンテーション画像を表示
    mask_img = Image.fromarray(segmentationMask * 255)
    mask_img = mask_img.resize(
        (targetWidth, targetHeight), Image.LANCZOS).convert("RGB")
    mask_img = tf.keras.preprocessing.image.img_to_array(
        mask_img, dtype=np.uint8)

    segmentationMask_inv = np.bitwise_not(mask_img)
    fg = np.bitwise_and(np.array(img), np.array(
        mask_img))
    '''plt.title('Foreground Segmentation')
    plt.imshow(fg)
    plt.show()'''
    bg = np.bitwise_and(np.array(img), np.array(
        segmentationMask_inv))
    '''plt.title('Background Segmentation')
    plt.imshow(bg)
    plt.show()'''

    human_segm = np.zeros((actualHeight,actualWidth,1),np.uint8)
    sum_heatmap = np.full((actualHeight,actualWidth,1),-1.0,np.float32)
    print("human_segm.shape="+str(human_segm.shape))
    # Part Heatmaps, PartOffsets,部位セグメンテーション画像を表示
    for i in range(partHeatmaps.shape[2]):

        heatmap = partHeatmaps[:, :, i]  # First Heat map
        heatmap[np.logical_not(tf.math.reduce_any(mask, axis=-1).numpy())] = -1
        # Set portions of heatmap where person is not present in segmentation mask, set value to -1

        # SHOW HEATMAPS

        '''plt.clf()
        plt.title('Heatmap: ' + PART_CHANNELS[i])
        plt.ylabel('y')
        plt.xlabel('x')
        print(type(heatmap * OutputStride))
        plt.imshow(heatmap * OutputStride)
        plt.show()'''
        
        actual_part = cv2.resize(heatmap, dsize = (actualWidth,actualHeight), interpolation = cv2.INTER_NEAREST)
        #actual_part = Image.fromarray(heatmap * OutputStride)

        '''print(type(actual_part))
        print(actual_part.shape)'''
        '''actual_part = actual_part.resize(
            (actualWidth, actualHeight), Image.LANCZOS)'''
        '''print(actual_part.dtype)
        print(heatmap[0][0])
        print(actual_part.getpixel((0,0)))
        print(actual_part[0][0])
        print(type(actual_part[0][0]))
        print(type(sum_heatmap[0][0]))'''
        for y in range(actualHeight):
            for x in range(actualWidth):
                if sum_heatmap[y][x] < actual_part[y][x]:
                    sum_heatmap[y][x] = actual_part[y][x]
                    if PART_CHANNELS[i] == "left_face" or PART_CHANNELS[i] == "right_face":
                        human_segm[y][x] = 1
                    elif PART_CHANNELS[i] == "left_upper_arm_front" or PART_CHANNELS[i] == "left_upper_arm_back":
                        human_segm[y][x] = 2
                    elif PART_CHANNELS[i] == "right_upper_arm_front" or PART_CHANNELS[i] == "right_upper_arm_back":
                        human_segm[y][x] = 3
                    elif PART_CHANNELS[i] == "left_lower_arm_front" or PART_CHANNELS[i] == "left_lower_arm_back":
                        human_segm[y][x] = 4
                    elif PART_CHANNELS[i] == "right_lower_arm_front" or PART_CHANNELS[i] == "right_lower_arm_back":
                        human_segm[y][x] = 5
                    elif PART_CHANNELS[i] == "left_hand":
                        human_segm[y][x] = 6
                    elif PART_CHANNELS[i] == "right_hand":
                        human_segm[y][x] = 7
                    elif PART_CHANNELS[i] == "torso_front" or PART_CHANNELS[i] == "torso_back":
                        human_segm[y][x] = 8
                    elif PART_CHANNELS[i] == "left_upper_leg_front" or PART_CHANNELS[i] == "left_upper_leg_back":
                        human_segm[y][x] = 9
                    elif PART_CHANNELS[i] == "right_upper_leg_front" or PART_CHANNELS[i] == "right_upper_leg_back":
                        human_segm[y][x] = 10
                    elif PART_CHANNELS[i] == "left_lower_leg_front" or PART_CHANNELS[i] == "left_lower_leg_back":
                        human_segm[y][x] = 11
                    elif PART_CHANNELS[i] == "right_lower_leg_front" or PART_CHANNELS[i] == "right_lower_leg_back":
                        human_segm[y][x] = 12
                    elif PART_CHANNELS[i] == "left_feet":
                        human_segm[y][x] = 13
                    elif PART_CHANNELS[i] == "right_feet":
                        human_segm[y][x] = 14
        '''plt.clf()
        plt.title('Heatmap: ' + PART_CHANNELS[i])
        plt.ylabel('y')
        plt.xlabel('x')
        plt.imshow(actual_part)
        plt.show()'''

        heatmap_sigmoid = tf.sigmoid(heatmap)
        y_heat, x_heat = np.unravel_index(
            np.argmax(heatmap_sigmoid, axis=None), heatmap_sigmoid.shape)

        partHeatmapPositions.append([x_heat, y_heat])
        partScores.append(heatmap_sigmoid[y_heat, x_heat].numpy())
        # Offset Corresponding to heatmap x and y
        x_offset = partOffsets[y_heat, x_heat, i]
        y_offset = partOffsets[y_heat, x_heat, partHeatmaps.shape[2]+i]
        partOffsetVector.append([x_offset, y_offset])

        key_x = x_heat * OutputStride + x_offset
        key_y = y_heat * OutputStride + y_offset
        partPositions.append([key_x, key_y])

    plt.clf()
    '''plt.title('sum_heatmap')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(sum_heatmap)
    plt.show()
    plt.title('human_segm')
    plt.imshow(human_segm)
    plt.show()'''
    human_segm,segm_contours_list = adjust_human_segm(human_segm)
    '''plt.title('human_segm_ver2')
    plt.imshow(human_segm)
    plt.show()'''
    #actualimg = cv2.imread(imagePath,-1)
    actualimg = cv2.resize(alpha_img, dsize = (actualWidth,actualHeight), interpolation = cv2.INTER_AREA)
    '''plt.title('actualimg')
    plt.imshow(cv2.cvtColor(actualimg,cv2.COLOR_BGRA2RGBA))
    plt.show()'''
    human_segm = adjust_actual_cut(actualimg,human_segm)
    '''plt.title('human_segm_ver3')
    plt.imshow(human_segm)
    plt.show()
    for y in range(actualHeight):
        for x in range(actualWidth):
            if human_segm[y][x] != 0:
                actualimg[y][x] = [human_segm[y][x],human_segm[y][x],human_segm[y][x],255]
    plt.title('actualimg_ver2')
    plt.imshow(cv2.cvtColor(actualimg,cv2.COLOR_BGRA2RGBA))
    plt.show()'''
    human_segm = adjust_actual_add(actualimg,human_segm,segm_contours_list)
    '''plt.title('human_segm_ver4')
    plt.imshow(human_segm)
    plt.show()
    for y in range(actualHeight):
        for x in range(actualWidth):
            if human_segm[y][x] != 0:
                actualimg[y][x] = [human_segm[y][x],human_segm[y][x],human_segm[y][x],255]
    plt.title('actualimg_ver3')
    plt.imshow(cv2.cvtColor(actualimg,cv2.COLOR_BGRA2RGBA))
    plt.show()'''
    human_segm = dilated_human_segm(human_segm,(3,3),ite=2)
    '''plt.title('human_segm_ver5')
    plt.imshow(human_segm)
    plt.show()
    cv2.imwrite(outdirPath+file_name,human_segm)'''#確認保存用

    print('partheatmapPositions', np.asarray(partHeatmapPositions).shape)
    print('partoffsetVector', np.asarray(partOffsetVector).shape)
    print('partkeypointPositions', np.asarray(partPositions).shape)
    print('partkeyScores', np.asarray(partScores).shape)

    return(actualimg,human_segm)


if __name__ == "__main__":
    segm_run()
