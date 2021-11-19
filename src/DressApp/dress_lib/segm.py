import numpy as np
from PIL import Image
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
import matplotlib.pyplot as plt

from PIL import ImageOps

import cv2
import gc

#人以外に検出してしまった領域を消す
def not_human_cut(size_tuple,trimap_img):
    trimap = trimap_img
    zero_img = np.zeros(size_tuple,np.uint8)
    _,cont_img = cv2.threshold(zero_img,10,255,cv2.THRESH_BINARY)#２値化する
    cont_img[trimap>0] = 255
    contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    maxs = [0,0]
    for i in range(len(contours)):
        #print(contours[i].shape)
        if maxs[0] < contours[i].shape[0]:
            maxs[0] = contours[i].shape[0]
            maxs[1] = i
    for i in range(len(contours)):
        if i != maxs[1]:
            cv2.drawContours(cont_img, contours, i, 0, -1)
    
    #plt.subplot(3,3,7)
    #plt.imshow(cont_img, cmap='gray', vmin=0, vmax=255)
    #plt.title("2num")
    #print(trimap[0][0])

    trimap[cont_img==0] = 0
    return trimap,cont_img

# 6.OpenCVで膨張収縮処理をしてtrimapを生成し、適当な場所に保存します。下の右のようなtrimapが得られます。
def gen_trimap(mask,k_size=(5,5),ite=1):
    h,w = mask.shape
    '''fig_def = plt.figure(figsize=(20,8))
    fig_def.suptitle("grey")'''

    kernel = np.ones(k_size,np.uint8) #要素が全て1の配列を生成
    eroded = cv2.erode(mask,kernel,iterations = ite)
    '''plt.subplot(1,4,2)
    plt.imshow(eroded,cmap='gray', vmin=0, vmax=255)
    plt.title("eroded")'''
    dilated = cv2.dilate(mask,kernel,iterations = ite)
    '''plt.subplot(1,4,3)
    plt.imshow(dilated,cmap='gray', vmin=0, vmax=255)
    plt.title("dilated")'''

    #print(dilated.dtype)
    trimap = np.full(mask.shape,128)
    trimap[eroded >= 1] = 255 #なぜか二値画像の白が12 dtypeを変更しても意味なし
    trimap[dilated == 0] = 0

    #残ってしまったメイン部分以外を消す
    trimap,niti = not_human_cut((h,w),trimap)
    '''plt.subplot(1,4,4)
    plt.imshow(trimap,cmap='gray', vmin=0, vmax=255)
    plt.title("no_human_cut")
    plt.show()'''

    return trimap,niti

def bokasi_filter(img):
    blur = cv2.blur(img,(15,15))
    return blur

def cutting_out():
    ctx = mx.cpu()

    #url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/voc_examples/1.jpg'
    #filename = gluoncv.utils.download(url)
    trimaps_dir = "./DressApp/dress_lib/indexnet_matting/examples/trimaps/"
    images_dir = "./DressApp/dress_lib/indexnet_matting/examples/images/"
    input_dir_name = "./DressApp/dress_lib/temporary_imgs/"
    downroad_root = "./DressApp/dress_lib/models"
    filename = "data1.jpg"
    filename = "Hito_risize.png"
    filename = "Hito_risize.jpg"
    #filename = "result.png"
    #filename = "Hito_risize_risize.jpg"

    #画像をNDArryで読み込む
    img = image.imread(input_dir_name+filename)
    print(type(img))

    #データの正規化
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(0).as_in_context(ctx)

    #モデルを読み込む
    #初回時に指定がなければ(default)/.mxnet/modelsに保存される
    #指定する場合にはrootで指定する
    #2回目以降はそこから読み込む
    model = gluoncv.model_zoo.get_model('deeplab_resnet152_voc', pretrained=True, root='./DressApp/dress_lib/models')
    model.collect_params().reset_ctx(ctx)

    #モデルの適応
    #人たけを抽出する（class:15）
    output = model.predict(img)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
    a = np.where(predict ==15, 255, 0).astype('int32')
    b = Image.fromarray(a).convert('L')


    #画像を改めてPILで読み込み、結果（b）と重ね合わせる
    img = Image.open(input_dir_name+filename)
    '''plt.imshow(img)
    plt.show()'''
    img.putalpha(b)
    '''plt.imshow(img)
    plt.show()'''
    #img.save('result.png')

    #mask作成
    #img_grey = cv2.imread('result.png',-1)
    img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGBA2BGRA)
    h,w,_ = img.shape
    trimap = np.zeros((h,w),np.uint8)
    #print(h)
    #print(w)
    #print(len(niti_img))
    for i in range(h):
        for j in range(w):
            if img[i][j][3] == 255:
                trimap[i][j] = 255


    #---以下から自作関数利用する

    #trimapを作成
    trimap,_ = gen_trimap(trimap,k_size=(5,5),ite=2)
    cv2.imwrite(trimaps_dir+'bler_'+filename.split(".")[0]+'.png',trimap)

    #blur画像を作成する
    img_copy = img.copy()
    h,w,_ = img_copy.shape
    for i in range(h):
        for j in range(w):
            img_copy[i][j][3] = 255
    img_copy = bokasi_filter(img_copy)
    for i in range(h):
        for j in range(w):
            if trimap[i][j] == 255:
                img_copy[i][j] = img[i][j]

    cv2.imwrite(images_dir+'bler_'+filename.split(".")[0]+'.png',cv2.cvtColor(img_copy,cv2.COLOR_BGRA2BGR))
    print("segm finish")
    del trimap
    del img
    del img_copy
    del _
    del output
    del predict
    del model
    del a
    del b
    del transform_fn

    gc.collect()