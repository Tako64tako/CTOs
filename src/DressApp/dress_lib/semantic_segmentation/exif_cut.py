from PIL import Image
from PIL import ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc
import os

#exif情報から画像を回転処理を行うモジュール
def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        print(exif_data)
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img
def load_image_file(file, mode='RGB'):#i画像ををexifを考慮してPILで読み込み
    # Load the image with PIL
    img = Image.open(file)

    # Exif データを取得
    # 存在しなければそのまま画像を返す
    try:
        exif = img._getexif()
        print("get_exif")
        if hasattr(ImageOps, 'exif_transpose'):
            # Very recent versions of PIL can do exit transpose internally
            img = ImageOps.exif_transpose(img)
        else:
            # Otherwise, do the exif transpose ourselves
            img = exif_transpose(img)

        #exif情報を消す
        print(4)
        data = img.getdata()
        print(5)
        mode = img.mode
        print(6)
        size = img.size
        print(7)
        img = Image.new(mode, size)
        print(8)
        img.putdata(data)
        print(9)
    except AttributeError:
        print("not_exif")

    return img

#圧縮する(pngのみ)モジュール
def imgEncodeDecode(in_imgs, ch, quality=5):
    """
    入力された画像リストを圧縮する
    [in]  in_imgs:  入力画像リスト
    [in]  ch:       出力画像リストのチャンネル数 （OpenCV形式）
    [in]  quality:  圧縮する品質 (1-100)
    [out] out_imgs: 出力画像リスト
    """

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    out_imgs = []

    for img in in_imgs:
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        if False == result:
            print('could not encode image!')
            exit()

        decimg = cv2.imdecode(encimg, ch)
        out_imgs.append(decimg)

    return out_imgs

def exifcut_compression_risize(cut_mode="haikei",filename="",input_img_path=""):
    print(0)
    #dir_name = "./DressApp/dress_lib/images/input_imgs/"
    output_dir = "./DressApp/dress_lib/images/temporary_imgs/"
    output_dir = "./DressApp/dress_lib/images/via/"#臨時追加
    #filename = "IMG_0137.png"
    print(1)

    #リサイズして画像の大きさを小さくする
    #hito = cv2.imread(output_dir+filename+".jpg")
    img = cv2.imread(input_img_path)
    img_format = filename.split('.')[1]

    h,w,_ = img.shape#cv2で読み込むとexif情報を解釈した画像配列作られるから、hとwが正常？
    print("h="+str(h))
    print("w="+str(w))
    scale_factor = 0
    if h > 800:
        print("do resize")
        print(800 / h)
        scale_factor = int(800 / h * 100) / 100
        print("scale_facter"+str(scale_factor))
        print(2)
        img = cv2.resize(img,dsize=None, fx=scale_factor, fy=scale_factor)
    h,w,_ = img.shape
    print("h="+str(h))
    print("w="+str(w))
    print(3)

    if img_format == "jpg":# もしjpg画像ならpngに変換
        print("this img is jpg")
        filename  = filename.split('.')[0]+".png"
        cv2.imwrite(output_dir+filename, img,[int(cv2.IMWRITE_PNG_COMPRESSION ), 1])
    else:
        print("this img is png")
        cv2.imwrite(output_dir+filename, img)#保存

    # exif情報をもとに画像を回転、exif情報削除した画像を生成
    img = load_image_file(output_dir+filename)#Pillowで開き、exif情報から回転させたものを格納し、exif情報を消した画像を生成
    #img.save("./resize_exif/"+filename)#保存
    img_size =os.path.getsize(output_dir+filename)
    print("img_size="+str(img_size))
    print("img_format="+str(img_format))

    if img_size > 500000:
        print("do compression")
        #jpgにして圧縮する
        img = np.array(img)
        print(10)
        img = img[...,::-1] #BGR->RGB
        print(11)
        _,_,ch = img.shape
        print(12)
        img = imgEncodeDecode([img], ch, 40)#jpg圧縮
        print(13)
        #cv2.imwrite(puress_dir+filename, img[0])#保存
        img = img[0]

        output_dir = "./DressApp/dress_lib/images/via/"#臨時追加
        cv2.imwrite(output_dir+filename, img,[int(cv2.IMWRITE_PNG_COMPRESSION ), 9])#確認保存用
        '''#pngで保存これはalphaのため -> 推論の際jpgでもpngにされるからjpgでもいい
        output_dir = "./DressApp/dress_lib/images/via/"#臨時追加
        if cut_mode == "haikei":
            cv2.imwrite(output_dir+filename, img,[int(cv2.IMWRITE_PNG_COMPRESSION ), 9])#確認保存用
        else:
            cv2.imwrite(output_dir+filename+'_risize.jpg', img)'''
    else:# 画像のデータサイズが小さいとき、圧縮しない
        print("don't compression")
        output_dir = "./DressApp/dress_lib/images/via/"#臨時追加
        img.save(output_dir+filename)#確認保存用
    
    print(14)
    del img
    print(15)
    gc.collect()
    print(16)
    return(output_dir,filename)
    '''img = load_image_file(input_img_path)#Pillowで開き、exif情報から回転させたものを格納

    #exif情報を消す
    print(2)
    data = img.getdata()
    print(3)
    mode = img.mode
    print(4)
    size = img.size
    print(5)
    img = Image.new(mode, size)
    print(6)
    img.putdata(data)
    print(7)

    #jpgにして圧縮する
    img = np.array(img)
    print(8)
    img = img[...,::-1] #BGR->RGB
    print(9)
    _,_,ch = img.shape
    print(10)
    img = imgEncodeDecode([img], ch, 40)#jpg圧縮
    print(11)
    #cv2.imwrite(output_dir+filename+'.jpg', img[0])#保存

    #リサイズして画像の大きさを小さくする
    #hito = cv2.imread(output_dir+filename+".jpg")
    h,w,_ = img[0].shape
    print(12)
    img = cv2.resize(img[0],dsize=None, fx=0.2, fy=0.2)
    print(13)

    #pngで保存これはalphaのため -> 推論の際jpgでもpngにされるからjpgでもいい
    output_dir = "./DressApp/dress_lib/images/via/"#臨時追加
    if cut_mode == "haikei":
        cv2.imwrite(output_dir+filename, img,[int(cv2.IMWRITE_PNG_COMPRESSION ), 9])#確認保存用
    else:
        cv2.imwrite(output_dir+filename+'_risize.jpg', img)
    print(14)
    del img
    print(15)
    gc.collect()
    print(16)
    return(output_dir)'''

#HEIC形式変更
#Exif情報を消す
#画像を圧縮する
#リサイズして元のサイズにする(画像を粗くする)
#画像のwidthとheightを小さくする(リサイズ)○