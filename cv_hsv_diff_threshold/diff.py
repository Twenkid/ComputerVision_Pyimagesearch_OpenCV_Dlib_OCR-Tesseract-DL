# 2017.12.22 15:48:03 CST
# 2017.12.22 16:00:14 CST
# masking: Silencer, https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
import cv2
import numpy as np


img1 = cv2.imread("D:\\py\\data\\1.png")
cv2.imshow("1", img1)
img2 = cv2.imread("D:\\py\\data\\2.png")
cv2.imshow("2", img2)
diff = cv2.absdiff(img1, img2)
mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

th = 1
imask =  mask>th

canvas = np.zeros_like(img2, np.uint8)
canvas[imask] = img2[imask]

#cv2.imwrite("result.png", canvas)
cv2.imshow("result.png", canvas)

cv2.waitKey(0)