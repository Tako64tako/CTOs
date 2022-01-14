# 各種パッケージをimportします。
import numpy as np
import cv2
import matplotlib.pyplot as plt

def cutting(blur_img,matte_img,filename):
    #画像、マスク(アルファチャンネル)を読み込み、背景を準備します。
    #blur_imgはndarray shapeは(h,w,3)
    img = blur_img
    img = img[...,::-1]
    matte = cv2.cvtColor(matte_img,cv2.COLOR_GRAY2BGR)

    print(matte.shape)
    h,w,_ = img.shape
    bg = np.full_like(img,255) #white background

    #マスクを0~1.に標準化し、画像、背景に下のようにそれぞれ掛けます。それらを足し合わせると最終的な出力が得られます。
    img = img.astype(float)
    bg = bg.astype(float)

    matte = matte.astype(float)/255
    img = cv2.multiply(img, matte)
    bg = cv2.multiply(bg, 1.0 - matte)
    outImage = cv2.add(img, bg)
    #plt.imshow(outImage/255)
    #plt.show()

    print(outImage[0][0])
    alpha_img = outImage.astype(np.uint8)
    print(alpha_img[0][0])

    # alpha画像にする ついでにmaskを作成する
    alpha_img = cv2.cvtColor(alpha_img,cv2.COLOR_RGB2BGRA)
    mask = np.full((h,w),255,np.uint8)
    print(alpha_img[0][0])
    h,w,_ = alpha_img.shape
    for y in range(h):
        for x in range(w):
            b,g,r,a = alpha_img[y][x]
            if b==255 and g ==255 and r==255:
                alpha_img[y][x] = (255,255,255,0)
                mask[y][x] = 0

    # 背景削除画像の補正処理　より正確に背景を切り抜く
    kernel = np.ones((10,10),np.uint8) #要素が全て1の配列を生成
    eroded = cv2.erode(mask,kernel,iterations = 1)
    mask[eroded==255] = 0
    gray_img = cv2.cvtColor(alpha_img,cv2.COLOR_BGRA2GRAY)
    for y in range(h):
        for x in range(w):
            if mask[y][x]==255 and gray_img[y][x] > 241:
                alpha_img[y][x][3] = 0
    
    #確認用
    '''plt.imshow(mask,cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.title("alpha_img")
    plt.imshow(cv2.cvtColor(alpha_img,cv2.COLOR_BGRA2RGBA))
    plt.show()'''
    return(alpha_img)

if __name__ == "__main__":
    cutting()