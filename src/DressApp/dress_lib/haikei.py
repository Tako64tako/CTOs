# 1.各種パッケージをimportします。今回はtorchvisionを使ってsegmentationを行います。
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


def inference(h,w,img):#推論を行う関数
    # 3.モデルをデバイスに渡し、推論モードに切り替えます。
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#gpu or cpuへの切り替え

    model_path = './DressApp/dress_lib/models'
    torch.hub.set_dir(model_path)
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
    model = model.to(device)
    model.eval()
    # 4.画像のnumpy配列をtensor型にし、正規化します。また、バッチの次元を追加します。
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    # 5.推論すると下の右のような画像が得られます。
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output = output.argmax(0)
    mask = output.byte().cpu().numpy()
    mask = cv2.resize(mask,(w,h))
    img = cv2.resize(img,(w,h))
    return img,mask

def bokasi_filter(img):
    blur = cv2.blur(img,(15,15))
    return blur

def bokasi_compound(img,trimap):
    blur_com = bokasi_filter(img)
    h,w,_ = img.shape
    for y in range(h):
        for x in range(w):
            if trimap[y][x]==255:
                blur_com[y][x] = img[y][x]
    return blur_com


#ヒストグラムを作成する。そして0以外の一番多い画素値のものだけ残して消す
def hist_cut(img,trimap_img):
    trimap = trimap_img.copy()
    hist = np.histogram(img,bins=np.arange(257))
    #print(hist)
    #print(len(hist[1]))
    max = 0
    max_num = 0
    for i in range(len(hist[0])):
        if i != 0 and max_num < hist[0][i]:
            max = i
            max_num = hist[0][i]
    #print(max)
    #print(max_num)
    trimap[img != max] = 0 #これで画素値を消す
    return trimap

#人以外に検出してしまった領域を消す
def not_human_cut(size_tuple,trimap):
    #print("not_human_cut is trimap = "+str(trimap.dtype))
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

    trimap[cont_img==0] = 0
    return trimap,cont_img

#bouding_boxに沿って切り抜く
def bounding_cut(img,trimap):
    #print("trimap of bouding_cut = "+str(trimap.dtype))
    cont_img = np.zeros(trimap.shape,np.uint8)#trimapはfloat64のため変換する必要がある
    cont_img[trimap >= 1] = 255
    contours, hierarchy = cv2.findContours(cont_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = cv2.boundingRect(contours[0])
    img = img[y:y+h,x:x+w]
    trimap = trimap[y:y+h,x:x+w]
    fig = plt.figure(figsize=(20,9))
    fig.suptitle("plot")
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title("part_img")
    plt.subplot(1,2,2)
    plt.imshow(trimap,cmap='gray', vmin=0, vmax=255)
    plt.title("part_trimap")
    plt.show()

    del cont_img
    del hierarchy
    return img,trimap

# 6.OpenCVで膨張収縮処理をしてtrimapを生成し、適当な場所に保存します。下の右のようなtrimapが得られます。
def gen_trimap(img,mask,k_size=(5,5),ite=1):
    h,w,_ = img.shape

    #ヒストグラムを作成する。そして0以外の一番多い画素値のものだけ残して消す
    mask = hist_cut(mask,mask)

    #残ってしまったメイン部分以外を消す
    trimap,niti = not_human_cut((h,w),mask)

    #ruslt用
    '''img = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
    for y in range(h):
        for x in range(w):
            if trimap[y][x] == 0:
                img[y][x][3] = 0'''

    kernel = np.ones(k_size,np.uint8) #要素が全て1の配列を生成
    eroded = cv2.erode(trimap,kernel,iterations = ite)
    dilated = cv2.dilate(trimap,kernel,iterations = ite)

    #print(dilated.dtype)
    trimap = np.full(mask.shape,128)#dtype=float64
    trimap[eroded >= 1] = 255 #なぜか二値画像の白が12 dtypeを変更しても意味なし
    trimap[dilated == 0] = 0

    img,trimap = bounding_cut(img,trimap)

    return img,trimap,niti

def cutting_out():
    # 2.画像を読み込み、DeepLabv3の入力サイズに合わせてリサイズします。
    image_path = 'IMG_0101.png'
    #image_path = 'data1.jpg'
    #image_path = 'kei.jpg'
    image_path = "Hito_iphassyu.jpg"
    image_path = "result1.png"
    image_path = "Hito_risize.jpg"
    image_path = "Hito_risize.png"
    #image_path = "model_0.png"

    trimaps_dir = "./DressApp/dress_lib/indexnet_matting/examples/trimaps/"
    images_dir = "./DressApp/dress_lib/indexnet_matting/examples/images/"
    input_dir_name = "./DressApp/dress_lib/temporary_imgs/"
    img = cv2.imread(input_dir_name+image_path)
    img = img[...,::-1] #BGR->RGB
    img_h,img_w,_ = img.shape #高さ 幅 色を代入
    real_img = img.copy()
    #img = cv2.resize(img,(320,320))
    img = cv2.resize(img,(img_w,img_h))
    img,mask = inference(img_h,img_w,img)#推論でmaskを生成する

    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img,trimap,niti = gen_trimap(img,mask,k_size=(3,3),ite=2)
    blur_img = bokasi_compound(img,trimap)
    cv2.imwrite(images_dir+"bler_"+image_path,blur_img)

    cv2.imwrite(trimaps_dir+"bler_"+image_path,trimap)
if __name__ == "__main__":
    cutting_out()
