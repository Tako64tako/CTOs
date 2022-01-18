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

# exif情報を持つ画像なら、exifに従って画像を回転補正したexif情報を持たない画像を生成する
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
        data = img.getdata()
        mode = img.mode
        size = img.size
        img = Image.new(mode, size)
        img.putdata(data)
    except AttributeError:
        print("not_exif")

    return img

#圧縮するモジュール
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

# リサイズして、フォーマット変換して、exif情報の処理をし、圧縮する
def exifcut_compression_risize(filename="",input_img_path="",output_dir=""):

    # リサイズして画像の大きさを小さくする
    img = cv2.imread(input_img_path)
    img_format = filename.split('.')[1]#入力された画像のフォーマットを得る
    h,w,_ = img.shape#cv2で読み込むとexif情報を解釈した画像配列が作られるから、hとwが正常？
    print("h="+str(h))
    print("w="+str(w))
    scale_factor = 0#リサイズ倍率
    #もしhが800より大きいならばリサイズする
    if h > 800:
        print("do resize")
        print(800 / h)
        scale_factor = int(800 / h * 100) / 100
        print("scale_facter"+str(scale_factor))
        img = cv2.resize(img,dsize=None, fx=scale_factor, fy=scale_factor)
    h,w,_ = img.shape
    print("h="+str(h))
    print("w="+str(w))
    #cv2.imwrite(output_dir+"risize.png",img)

    
    # 画像のフォーマットをpngにする
    print("img_format="+str(img_format))
    if img_format == "jpg" or img_format == "JPG":# もしjpg画像ならpngに変換
        print("this img is jpg")
        filename  = filename.split('.')[0]+".png"
        #cv2.imwrite(output_dir+"format_change.png", img,[int(cv2.IMWRITE_PNG_COMPRESSION ), 1])
        cv2.imwrite(output_dir+filename, img,[int(cv2.IMWRITE_PNG_COMPRESSION ), 1])
    elif img_format == "png":
        print("this img is png")
        cv2.imwrite(output_dir+filename, img)#保存
        #cv2.imwrite(output_dir+"format_change.png", img)
    else:
        print("this format isn't supported")


    # exif情報をもとに画像を回転、exif情報を削除した画像を生成
    img = load_image_file(output_dir+filename)#Pillowで開き、exif情報から回転させたものを格納し、exif情報を消した画像を生成
    #img.save(output_dir+"exifcut.png")#保存


    # 画像のデータサイズが500kBより大きいなら圧縮しておく
    img_size =os.path.getsize(output_dir+filename)
    print("img_size="+str(img_size))
    if img_size > 500000:
        print("do compression")
        #jpgにして圧縮する
        img = np.array(img)
        img = img[...,::-1] #BGR->RGB
        _,_,ch = img.shape
        img = imgEncodeDecode([img], ch, 40)#jpg圧縮
        #cv2.imwrite(output_dir+"puress.png", img[0])#保存
        img = img[0]

        #pngで保存
        cv2.imwrite(output_dir+filename, img,[int(cv2.IMWRITE_PNG_COMPRESSION ), 9])#確認保存用
    else:# 画像のデータサイズが小さいとき、圧縮しない
        print("don't compression")
        #pngで保存
        img.save(output_dir+filename)#確認保存用
    
    del img
    gc.collect()
    return(filename)
