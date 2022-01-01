import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    human_segm = cv2.imread("./materials/part_segms/IMG_0138.png",0)
    plt.figure(1)
    plt.title('Segmentation Mask')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(human_segm)

    skeleton_img = cv2.imread("./images/skeleton_images/IMG_0138.png")
    plt.figure(2)
    plt.title('skeleton_img')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(cv2.cvtColor(skeleton_img,cv2.COLOR_BGR2RGB))

    '''brank_img = cv2.imread("./images/temporary_imgs/brank.png",-1)
    plt.figure(3)
    plt.title('brank_img')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(cv2.cvtColor(brank_img,cv2.COLOR_BGRA2RGBA))

    boding_img = cv2.imread("./images/temporary_imgs/boding.png",0)
    plt.figure(4)
    plt.title('boding_img')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.imshow(boding_img)'''
    print("yes")
    plt.show()
