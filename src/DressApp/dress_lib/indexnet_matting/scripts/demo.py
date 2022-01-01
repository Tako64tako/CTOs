
import numpy as np
import os
import cv2 as cv
from time import time
from PIL import Image
from PIL import ImageOps
from collections import OrderedDict

import torch
import torch.nn as nn
from .hlmobilenetv2 import hlmobilenetv2

import matplotlib.pyplot as plt

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

IMG_SCALE = 1./255
IMG_MEAN = np.array([0.485, 0.456, 0.406, 0]).reshape((1, 1, 4))
IMG_STD = np.array([0.229, 0.224, 0.225, 1]).reshape((1, 1, 4))

STRIDE = 32
RESTORE_FROM = './DressApp/dress_lib/indexnet_matting/pretrained/indexnet_matting.pth.tar'
#RESULT_DIR = './DressApp/dress_lib/images/mattes'
RESULT_DIR = "./DressApp/dress_lib/images/via/"#臨時追加

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# load pretrained model
net = hlmobilenetv2(
        pretrained=False,
        freeze_bn=True, 
        output_stride=STRIDE,
        apply_aspp=True,
        conv_operator='std_conv',
        decoder='indexnet',
        decoder_kernel_size=5,
        indexnet='depthwise',
        index_mode='m2o',
        use_nonlinear=True,
        use_context=True
    )

try:
    checkpoint = torch.load(RESTORE_FROM, map_location=device)
    pretrained_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        if 'module' in key:
            key = key[7:]
        pretrained_dict[key] = value
except:
    raise Exception('Please download the pretrained model!')
net.load_state_dict(pretrained_dict)
net.to(device)
if torch.cuda.is_available():
    net = nn.DataParallel(net)

# switch to eval mode
net.eval()

def read_image(x):
    img_arr = np.array(Image.open(x))
    print(img_arr.shape)
    return img_arr

def image_alignment(x, output_stride, odd=False):
    imsize = np.asarray(x.shape[:2], dtype=np.float)
    if odd:
        new_imsize = np.ceil(imsize / output_stride) * output_stride + 1
    else:
        new_imsize = np.ceil(imsize / output_stride) * output_stride
    h, w = int(new_imsize[0]), int(new_imsize[1])

    x1 = x[:, :, 0:3]
    x2 = x[:, :, 3]
    new_x1 = cv.resize(x1, dsize=(w,h), interpolation=cv.INTER_CUBIC)
    new_x2 = cv.resize(x2, dsize=(w,h), interpolation=cv.INTER_NEAREST)

    new_x2 = np.expand_dims(new_x2, axis=2)
    new_x = np.concatenate((new_x1, new_x2), axis=2)

    return new_x

def inference(filename, image, trimap):
    with torch.no_grad():
        #image, trimap = read_image(image_path), read_image(trimap_path)
        trimap = np.expand_dims(trimap, axis=2)
        #print(image[0][0])
        #print(trimap[0][0])
        #plt.imshow(image)
        #plt.show()
        #plt.imshow(trimap)
        #plt.show()
        image = np.concatenate((image, trimap), axis=2)
        
        h, w = image.shape[:2]

        image = image.astype('float32')
        image = (IMG_SCALE * image - IMG_MEAN) / IMG_STD
        image = image.astype('float32')

        image = image_alignment(image, STRIDE)
        inputs = torch.from_numpy(np.expand_dims(image.transpose(2, 0, 1), axis=0))
        inputs = inputs.to(device)
        
        # inference
        start = time()
        print(2)
        outputs = net(inputs)
        end = time()

        outputs = outputs.squeeze().cpu().numpy()
        alpha = cv.resize(outputs, dsize=(w,h), interpolation=cv.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) * 255.
        trimap = trimap.squeeze()
        mask = np.equal(trimap, 128).astype(np.float32)
        alpha = (1 - mask) * trimap + mask * alpha

        #_, image_name = os.path.split(image_path)
        #Image.fromarray(alpha.astype(np.uint8)).save(os.path.join(RESULT_DIR, filename))#確認保存用
        # Image.fromarray(alpha.astype(np.uint8)).show()

        running_frame_rate = 1 * float(1 / (end - start)) # batch_size = 1
        print('framerate: {0:.2f}Hz'.format(running_frame_rate))

        print(type(alpha.astype(np.uint8)))
        return(alpha.astype(np.uint8))


def infer(blur_img,trimap_img,filename):
    image_path = [
        #'./examples/images/beach-747750_1280_2.png',
        #'./examples/images/boy-1518482_1920_9.png',
        #'./examples/images/light-bulb-1104515_1280_3.png',
        #'./examples/images/spring-289527_1920_15.png',
        #'./examples/images/wedding-dresses-1486260_1280_3.png',
        #"./examples/images/kei.jpg",
        #"./examples/images/man.png",
        #"./examples/images/data1.jpg",
        #"./examples/images/Hito_risize.jpg",
        #"./examples/images/masut.png",
        #"./examples/images/bler_kei.png",
        #"./DressApp/dress_lib/indexnet_matting/examples/images/bler_Hito_risize.png",
        "./DressApp/dress_lib/images/images/bler_IMG_0137_risize.png",
        #"./DressApp/dress_lib/indexnet_matting/examples/images/bler_Hito_risize_risize.png",
        #"./examples/images/Hito.png"
    ]
    trimap_path = [
        #'./examples/trimaps/beach-747750_1280_2.png',
        #'./examples/trimaps/boy-1518482_1920_9.png',
        #'./examples/trimaps/light-bulb-1104515_1280_3.png',
        #'./examples/trimaps/spring-289527_1920_15.png',
        #'./examples/trimaps/wedding-dresses-1486260_1280_3.png',
        #"./examples/trimaps/kei.jpg",
        #"./examples/trimaps/man.png",
        #"./examples/trimaps/data1.jpg",
        #"./examples/trimaps/Hito_risize.jpg",
        #"./examples/trimaps/masut.png",
        #"./examples/trimaps/bler_kei.png",
        #"./DressApp/dress_lib/indexnet_matting/examples/trimaps/bler_Hito_risize.png",
        "./DressApp/dress_lib/images/trimaps/bler_IMG_0137_risize.png",
        #"./DressApp/dress_lib/indexnet_matting/examples/trimaps/bler_Hito_risize_risize.png",
        #"./examples/trimaps/Hito.png"
    ]
    print(1)
    imgs = [blur_img]
    trimaps = [trimap_img]
    filenames = [filename]
    #for image, trimap in zip(image_path, trimap_path):
    for filename, image, trimap in zip(filenames, imgs, trimaps):
        matte = inference(filename, image, trimap)
        print("last")
    return(matte)

