import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

image_path = 'deta1.jpg'

img_plt = Image.open("./examples/images/"+image_path)


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
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
def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = Image.open(file)

    if hasattr(ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    #img = img.convert(mode)

    return np.array(img)

'''
convert_image = {
  1: lambda img: img,
  2: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),                              # 左右反転
  3: lambda img: img.transpose(Image.ROTATE_180),                                   # 180度回転
  4: lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),                              # 上下反転
  5: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90),  # 左右反転＆反時計回りに90度回転
  6: lambda img: img.transpose(Image.ROTATE_270),                                   # 反時計回りに270度回転
  7: lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270), # 左右反転＆反時計回りに270度回転
  8: lambda img: img.transpose(Image.ROTATE_90),                                    # 反時計回りに90度回転
}
try:
    img_exif = img_plt._getexif()
    print(img_exif)
    if img_exif:
        #orientation = img_exif.get(0x112, 1)
        #print(orientation)
        #img = convert_image[orientation](img_plt)
        Image.ImageOps.exif_transpose(img_plt)
except AttributeError:
    print(img_plt)'''

#img, trimap = np.array(img_plt), np.array(Image.open("./examples/trimaps/"+image_path))
img, trimap = load_image_file(file="./examples/images/"+image_path), load_image_file("./examples/trimaps/"+image_path)
trimap = np.expand_dims(trimap, axis=2)
#img = cv2.imread("./examples/images/"+image_path)
#img = img[...,::-1]
#trimap = cv2.imread("./examples/trimaps/"+image_path,0)
#trimap = cv2.imread("./examples/trimaps/"+"kei.jpg")
plt.gray()
plt.imshow(img)
plt.show()
plt.imshow(trimap)
plt.show()

print(img[0][0])
print(trimap[0][0])
image = np.concatenate((img, trimap), axis=2)

print(image)
#plt.imshow(image)
#plt.show()