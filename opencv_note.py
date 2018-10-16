1、获取hsv图像中的某个像素范围内掩码，掩码是理解就是一个8位单通道图像（灰度图、二值图），每个像素点要么是0要么是255，0不起作用
mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
res = cv2.bitwise_and(src_seg, src_seg, mask=mask) 可以获取掩码处的原图，注意mask只能是对应src_seg的掩码，大小通道要一致

2、判断图片模糊情况
import cv2

#https://www.jianshu.com/p/b0fa7a8eba78
def getImageVar(imgPath):
    image = cv2.imread(imgPath);
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar