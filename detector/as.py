# # This will import Image and ImageEnhance modules
from PIL import Image, ImageStat, ImageEnhance
import numpy
import math
import cv2
import matplotlib.pyplot as plt
#Opening Image
im = Image.open(r"C:\Users\tranc\Desktop\simple-blink-detector-master\detector\unnamed.jpg")

im3 = ImageEnhance.Contrast(im)
im3.enhance(2).show()
# Creating object of Brightness class
# g

#
# # showing resultant image
# im3.enhance(10).show()
im3 = im3.enhance(2).save(r"C:\Users\tranc\Desktop\simple-blink-detector-master\detector\chinh1.PNG")
# def brightness1( im_file ):
#    im = Image.open(im_file)
#    stat = ImageStat.Stat(im)
#    r = stat.mean[0]
#    g = stat.mean[1]
#    b = stat.mean[2]
#
#    return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
# #
# print(brightness1(r"C:\Users\tranc\Desktop\simple-blink-detector-master\detector\chinh1.PNG"))
# img = cv2.imread(r"C:\Users\tranc\Desktop\simple-blink-detector-master\detector\unnamed.jpg")
#
# img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#
# # equalize the histogram of the Y channel
# img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
#
# # convert the YUV image back to RGB format
# img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
#
# cv2.imshow('Color input image', img)
# cv2.imshow('Histogram equalized', img_output)
#
# cv2.waitKey(0)