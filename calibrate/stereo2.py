import numpy as np
import cv2
from matplotlib import pyplot as plt
# Stereo2: 24.8.2021
# Histogram equalization, CLAHE, ... to compensate - can't adjust both cameras, one of them is always automatic?
# Logitech C270
# Bad results so far

bCam = False
bFirst = True
bCam = True
if bFirst and not bCam:
    capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.waitKey(2000)
    ret, imgL = capL.read()
    cv2.imwrite("left.jpg", imgL);    
    capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("Press key...")
    cv2.waitKey(2000)
    ret, imgR = capR.read()
    cv2.imwrite("right.jpg", imgL);
    print("Check the files...")
    capL.release()
    capR.release()  


imgL = cv2.imread('left.jpg',0)
imgR = cv2.imread('right.jpg',0)

print("Press ESC for exit...")
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
Eq = 2 #CLAHE, 1 = equ

if bCam:
  capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  cv2.waitKey(1000)
  capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)
  cv2.waitKey(1000)
  while True:
      ret,imgL = capL.read()
      print("Press a key...")
      k = cv2.waitKey(1)
      ret,imgR = capR.read()
      if not ret:
        k = cv2.waitKey(33)
        kk = k and 0xff
        if kk == 0x27: break
        continue
      imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
      imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
      #imgL = cv2.normalize(imgL, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      #imgR = cv2.normalize(imgR, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      
      #imgL = cv2.normalize(imgL, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      #imgR = cv2.normalize(imgR, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      if Eq==1:
        imgL = cv2.equalizeHist(imgL)
        imgR = cv2.equalizeHist(imgR)
      elif Eq==2:
        clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
        imgL = clahe.apply(imgL)
        imgR = clahe.apply(imgR)
          
      #imgL.convertTo(imgL, cv2.CV_8UC1)
      #imgR.convertTo(imgR, cv2.CV_8UC1)
      cv2.imshow("L", imgL)
      cv2.imshow("R", imgR)
      k = cv2.waitKey(1)
      
      disparity = stereo.compute(imgL,imgR)
      #cv2.imshow("DISPARITY", disparity) #gray
      norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      cv2.imshow("DISPARITY_NORM", norm_image) #gray
      #plt.imshow(disparity,'gray') #OK
      plt.show()
      k = cv2.waitKey(33)
      kk = k and 0xff
      if kk == 0x27: break
  
"""
#num = 10
#found = 0
#w = 0; h = 0
#w, h, ch = ...
"""

if bCam:
  ret, imgL = capL.read()
  print(ret, "imgL")
  ret, imgR = capR.read()
  print(ret, "imgR")

cv2.imshow("L", imgL)
cv2.imshow("R", imgR)
cv2.waitKey(0)
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
cv2.imshow("L", imgL)
cv2.imshow("R", imgR)
cv2.waitKey(0)
capL.release()
capR.release()  


#while(found < num):  # Here, 10 can be changed to whatever number you like to choose  
#    ret, img = cap.read() # Capture frame-by-frame
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#    h,w,ch = img.shape    
    
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()