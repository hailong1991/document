1、获取hsv图像中的某个像素范围内掩码，掩码是理解就是一个8位单通道图像（灰度图、二值图），每个像素点要么是0要么是255，0不起作用
mask = cv2.inRange(hsv,color_dict[d][0],color_dict[d][1])
res = cv2.bitwise_and(src_seg, src_seg, mask=mask) 可以获取掩码处的原图，注意mask只能是对应src_seg的掩码，大小通道要一致
