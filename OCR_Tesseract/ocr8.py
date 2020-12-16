# Code for the Stackoverflow thread: https://stackoverflow.com/questions/65277845/how-to-obtain-the-best-result-from-pytesseract/65278387?noredirect=1#comment115407021_65278387
# Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
# python -m pip install pytesseract
# Languages: https://askubuntu.com/questions/793634/how-do-i-install-a-new-language-pack-for-tesseract-on-16-04
# Question and basic recognition code by Matteo
# Extended with additional image processin, correction of the output and comments
# by Todor Arnaudov, 13.12.2020 - 14.12.2020 - 16.12
# Recognizes "Cucine" if given only with the box with "Lube"
# Kitchen - attempts
# Recognizes all except "O" in CREO, need more transforms of the low-contrast interrupted last "O"
# See also: https://medium.com/better-programming/beginners-guide-to-tesseract-ocr-using-python-10ecbb426c3d
# https://github.com/tesseract-ocr/tesseract/wiki
# for configuration
# ocr8.py, logou.png
# CUCINE
#
# LUBE
#
# STORE
#
# PRATOLA PELIGNA
#
# CRE
#
# KITCHENS


import cv2
import numpy as np
import pytesseract


#pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\pytesseract\tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print(pytesseract.pytesseract.tesseract_cmd)

def Correct(text):
  Replace = { "§":"S", " ":"", "&":""} #replace also other non-letter characters, possibly digits
  corrected = text
  for k in Replace:
    print(k)
    corrected = corrected.replace(k,Replace[k])
  return corrected
 
#SimilarA = ["N","S","S", "8"]
#SimilarB = ["W","&", §
Similar = { "N":"W", "W":"N", "§":"S", "S":"§", "§":"&", "&":"§", "8":"&", "&":"8", "1":"l"}
# Should check both for lower case and for capital (lower L is similar to 1)
# Use a better structure of correspondence. E.g. one symbol could be similar to many others
# After recognizing - show similar variants, compare to dictionary etc.
   

path_to_image = "logo.png"
path_to_image = "logo2.png" #cropped the region with "Cucine" and "Lube"
path_to_image = "kitchens.png"
path_to_image = "logou.png" # Whitened the right side of "U" of "Cucine", from 200 to 255, in order to correct the gradient on the right, yet keep the "U" correct. Without that the "U" starts to resemble "L" and is recognized as "CULCINE"
# Could be automated with selective filtering - thicken lines like that

image = cv2.imread(path_to_image)
h, w, _ = image.shape    

def Contrast(image, imageOut, contrast, bright):
    #Change contrast and brightness - might be not necessary, just find appropriate values for the mask
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                imageOut[y,x,c] = np.clip(contrast*image[y,x,c] + bright, 0, 255)
    cv2.imshow('Contrast-ImageOut', imageOut)
    cv2.waitKey(0)    


def Partition(image):
  '''
  Partition to regions in order to process one-by-one.
  Then traverse the contours, check their sizes, select the big one (to some measure),
  possibly apply again and find sub regions. In current case the first pass partitions
  the left and right blocks.
  If not drawing the rectangle, it crops the LUBE and CREO blocks.
  '''
  h, w, _ = image.shape  
  cv2.rectangle(image,(0,0),(w,h),255,2)  #used in order to secure at least one region - the whole image
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  cnts, hier = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cropped = []
  for i, cnt in enumerate(cnts):
      x,y,w,h = cv2.boundingRect(cnt)
      #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
      cropped.append(image[y:y+h,x:x+w])
      cv2.imshow("CONTOUR: "+str(i), image[y:y+h,x:x+w])
    
    
Partition(image.copy())     
imageContrast = image.copy()
#Contrast(image,imageContrast , 1.5, 1.0)  #no; slowdd
custom_config = r'--oem 3 --psm 12'  #--psm 13 RAW_LINE 12 SPARSE_TEXT orientation and script detection  11 - as much as possible text
text = pytesseract.image_to_string(imageContrast, config=custom_config)#, lang="ita")
print(text)
cv2.waitKey(0)  

        

#h, w, _ = image.shape
h1, w1 = h, w
w*=5; h*=5
w = (int)(w); h = (int) (h)
#green channel

alpha = 1.4 #contrast
beta = -0.4 #brightness

#alpha = 1.0 #contrast
#beta =  0.0 #brightness

image2 = image.copy()
#image2 = cv2.resize(image2,(w*4,h*4), interpolation = cv2.INTER_AREA)
#image2 = cv2.resize(image2,(w*2,h*4), interpolation = cv2.INTER_AREA)
#cv2.imshow("SHEAR",image2)

b,g,r = cv2.split(image2)

'''
gLines = g.copy()

gLines = cv2.Canny(gLines,50,150,apertureSize = 7)

#HoughLines:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
lines = cv2.HoughLines(gLines,1,np.pi/180,3)
for rho,theta in lines[0]:
    a1 = np.cos(theta)
    b1 = np.sin(theta)
    x0 = a1*rho
    y0 = b1*rho
    x1 = int(x0 + 1000*(-b1))
    y1 = int(y0 + 1000*(a1))
    x2 = int(x0 - 1000*(-b1))
    y2 = int(y0 - 1000*(a1))

    cv2.line(gLines,(x1,y1),(x2,y2),(255,255,255),2)

cv2.imwrite('houghlines3.jpg',gLines)
'''

kernel1 = np.ones((1, 1), np.uint8)  
kernel3 = np.ones((3, 3), np.uint8) 
# Using cv2.erode() method  
#contoursBW = cv2.dilate(contoursBW, kernel)  
g = cv2.dilate(g, kernel1) #cv2.GaussianBlur(contoursBW,(1,1),0)
#contoursBW = cv2.erode(contoursBW, kernel, 1)
#contoursBW = cv2.GaussianBlur(contoursBW,(5,5),0)

mask = cv2.inRange(g, 210, 255) 
b[mask!=0] = 0
g[mask!=0] = 0
r[mask!=0] = 0

b[mask==0] = 255
g[mask==0] = 255
r[mask==0] = 255
cv2.imshow("G",g)
#g = cv2.GaussianBlur(g,(1,1),0)
cv2.imshow("G GAUSIIAN",g)

custom_config = r'--oem 3 --psm 12'  #--psm 13 RAW_LINE-
text = pytesseract.image_to_string(g, config=custom_config)#, lang="ita")
box = pytesseract.image_to_boxes(g, config=custom_config)
print(text)
print(box)
# RECOGNIZES all but the "O" in "CREO"
# Parse data: ... #16-12-2020
# Find lines (CUCINE, LUBE...) and test for single letter before and after,
# box size = average size of the previous letters, possibly scaled a bit 
'''
C 39 211 47 221 0
U 62 211 69 221 0
C 84 211 92 221 0
I 107 211 108 221 0
N 123 211 131 221 0
E 146 211 153 221 0
L 39 108 59 166 0
U 63 107 93 166 0
B 98 108 128 166 0
E 133 108 152 166 0
S 440 134 468 173 0
T 470 135 499 173 0
O 500 134 539 174 0
R 544 135 575 173 0
E 580 135 608 173 0
P 287 76 315 114 0
R 319 76 350 114 0
A 352 76 390 114 0
T 387 76 417 114 0
O 417 75 456 115 0
L 461 76 487 114 0
A 489 76 526 114 0
P 543 76 572 114 0
E 576 76 604 114 0
L 609 76 634 114 0
I 639 76 643 114 0
G 649 75 683 115 0
N 690 76 722 114 0
A 726 76 764 114 0
C 21 30 55 65 0
R 62 31 93 64 0
E 99 31 127 64 0
K 47 19 52 25 0
I 61 19 62 25 0
T 71 19 76 25 0
C 84 19 89 25 0
H 96 19 109 25 0
E 113 19 117 25 0
N 127 19 132 25 0
S 141 19 145 22 0
'''


cv2.waitKey(0)  

#g = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,1)

#g = cv2.dilate(g, kernel3)
g2 = g.copy()

# Search for circles. Goal: find the "O" of CREO and draw it as a circle, then recognize it.
# However it needs adjustments - other circles are also discovered.
circles = cv2.HoughCircles(g2,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=10,maxRadius=50)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    #cv2.circle(g,(i[0],i[1]),i[2],(0,255,0),2)
    # Draw the center of the circle
    cv2.circle(g2,(i[0],i[1]),2,(0,0,255),2)

cv2.imshow('detected circles',g2)
text = pytesseract.image_to_string(g2, config=custom_config)#, lang="ita")
print(text)
cv2.waitKey(0)  


contrast = image.copy()
'''
#Change contrast and brightness - might be not necessary, just find appropriate values for the mask
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
            contrast[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
cv2.imshow('contrast', contrast)
cv2.waitKey(0)            
'''
h, w, _ = contrast.shape 
#contrast = cv2.resize(contrast,(w*5,h*5), interpolation = cv2.INTER_AREA)
#contrast = cv2.resize(contrast,(w*4,h*4), interpolation = cv2.INTER_AREA)

#contrastBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contrastBW = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
contoursBW = contrastBW.copy()
text = pytesseract.image_to_string(contrastBW, config=custom_config)#psm 13 -- line, lang="ita")
print(text)
custom_config_8 = r'--oem 3 --psm 8' #SINGLE_WORD
text = pytesseract.image_to_string(contrastBW, config=custom_config_8)#psm 13 -- line, lang="ita")
print(text)
#corrected = text.lower()
#corrected = text.upper()
Replace = { "§":"S", " ":"" }
corrected = text
for k in Replace:
  print(k)
  corrected = corrected.replace(k,Replace[k])
print("CORRECTED:", corrected)
cv2.imshow('contrastBW', contrastBW)

#"KitcheWs" --> check a dictionary, similarity to real words (possibly test with normalized forms,
# lemmatized, if ending with an "s", test without also etc.
#English dictionary... from NLTK, WordNet or whatever
Vocab = ["kitchen"]  #...

cv2.waitKey(0)
cv2.floodFill(contrastBW, None, (0,0), 255)
contrastBW = cv2.adaptiveThreshold(contrastBW, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,3,1)


cv2.imshow('contrastBW', contrastBW)
cv2.waitKey(0)  
text = pytesseract.image_to_string(contrastBW)#, lang="fita")
print(text)

contoursBW = cv2.Canny(contoursBW, 30, 200)
#contoursBW = cv2.medianBlur(contoursBW,3)
kernel2 = np.ones((2, 2), np.uint8)  
kernel1 = np.ones((1, 1), np.uint8)  
kernel3 = np.ones((3, 3), np.uint8) 
# Using cv2.erode() method  
#contoursBW = cv2.dilate(contoursBW, kernel)  
contoursBW = cv2.GaussianBlur(contoursBW,(1,1),0)
#contoursBW = cv2.erode(contoursBW, kernel, 1)
#contoursBW = cv2.GaussianBlur(contoursBW,(5,5),0)
contoursBW = cv2.dilate(contoursBW, kernel2)  
contoursBW = cv2.erode(contoursBW, kernel1, 1)

#kernel1 = np.ones((3, 3), np.uint8)  
#contoursBW = cv2.morphologyEx(contoursBW, cv2.MORPH_OPEN, kernel3)

#cnts, hier = cv2.findContours(contoursBW, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#drawContoursImage = np.empty_like(contoursBW)
#cv2.drawContours(drawContoursImage, cnts, -1, 255, 2)
text = pytesseract.image_to_string(contoursBW,lang="eng") #contrastBW)#, lang="ita")
print(text)
cv2.imshow('drawContoursImage', contoursBW) #drawContoursImage)
cv2.waitKey(0) 

image = contrast

b,g,r = cv2.split(image)
mask = cv2.inRange(g, 129, 255) 
b[mask!=0] = 0
g[mask!=0] = 0
r[mask!=0] = 0

b[mask==0] = 255
g[mask==0] = 255
r[mask==0] = 255

#image = cv2.merge([b,g,r])
cv2.imshow('MASK', g)
cv2.waitKey(0)
g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('MASK', g)
#image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)

#Colors are not needed, but it doesn't matter
text = pytesseract.image_to_string(g, config=custom_config)#, lang="ita")
print("MASK:", text)
corrected = Correct(text)
print("CORRECTED: ", corrected)
cv2.waitKey(0)
#text = pytesseract.image_to_string(g, lang="eng",  config=custom_config)
#print(text) 
#cv2.waitKey(0)

cv2.imwrite("logo_without_threshold_median_etc.png", g)

def Additional():
    image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA) #Resize 3 times
    # converting image into gray scale image
    #mask = mas

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grey image', gray_image)
    cv2.waitKey(0)
    # converting it to binary image by Thresholding
    # this step is require if you have colored image because if you skip this part
    # then tesseract won't able to detect text correctly and this will give incorrect result
    threshold_1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # display image
    cv2.imshow('threshold image default', threshold_1)            
    cv2.imwrite("logo_thresh_bin.png", threshold_1)
    cv2.waitKey(0)

    threshold_img = threshold_1

    #cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    threshold_img = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,            cv2.THRESH_BINARY,13,2) #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)[1]
    #cv2.imshow('ADAPTIVE THRESHOLD image', threshold_img)    

    threshold_img = cv2.resize(threshold_img, (w1,h1), interpolation = cv2.INTER_AREA) #Resize 3 times

    #cv2.waitKey(0)
    #threshold_img = cv2.GaussianBlur(threshold_img,(3,3),0)
    #threshold_img = cv2.GaussianBlur(threshold_img,(3,3),0)
    #threshold_img = cv2.medianBlur(threshold_img,3)
    #cv2.imshow('medianBlur', threshold_img)            
    #cv2.waitKey(0)
    threshold_img  = cv2.bitwise_not(threshold_img)
    cv2.imshow('Invert', threshold_img)            
    cv2.waitKey(0)
    cv2.imwrite("logo_out.png", threshold_img)
    kernel = np.ones((1, 1), np.uint8)  
    # Using cv2.erode() method  
    threshold_img = cv2.dilate(threshold_img, kernel)  
    cv2.imshow('Dilate', threshold_img)            
    cv2.waitKey(0)
    #cv2.imshow('threshold image', threshold_img)
    # Maintain output window until user presses a key
    #cv2.waitKey(0)
    # Destroying present windows on screen
    cv2.destroyAllWindows()
    # now feeding image to tesseract
    print(dir(pytesseract))
    text = pytesseract.image_to_string(threshold_img, config=custom_config) #lang="ita")
    box = pytesseract.image_to_boxes(threshold_img, config=custom_config) #lang="ita")
    data = pytesseract.image_to_data(threshold_img, config=custom_config) #lang="ita")
    print(text)
    print(box)
    print(data)
    
#Additional()    #trying other transformations
