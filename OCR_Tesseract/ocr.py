# USAGE
# python ocr.py --image images/example_01.png 
# python ocr.py --image images/example_02.png  --preprocess blur
#+ 12-7-2017 + Todor
# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
#from pytesseract import tesseract_cmd

# Todor Arnaudov:
# In case of error messages, find the library source file and set the path to the Tesseract.exe:
# D:\Py\pyimage\ocr\ocr.py
# Run as administrator:
# "C:\Program Files\Python36\Lib\site-packages\pytesseract\pytesseract.py"
# Set the path: ...
#pytesseract.tesseract_cmd = r"D:\OCR\tesseract.exe" #r"E:\Tesseract\tesseract.exe" #imdisk RAM disk
#not set???


#s = dir(pytesseract)
#print(s)

#pytesseract.set_path(r"E:\Tesseract\tesseract.exe" )

#pytesseract.set_path( r"D:\OCR\tesseract.exe" )


#print(s)

#print(pytesseract.tesseract_cmd)
#pytesseract.tesseract_cmd= r"D:\OCR\tesseract.exe" 
#print(pytesseract.tesseract_cmd)
# construct the argument parse and parse the arguments

# -i image.jpg ... -l bul|eng|jpn|chi  (?) - see in D:\OCR\tessdata 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image to be OCR'd")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
	help="type of preprocessing to be done")

ap.add_argument("-l", "--lang", type=str, default="eng")


args = vars(ap.parse_args())

# load the example image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lang = args["lang"]

cv2.imshow("Image", gray)

# check to see if we should apply thresholding to preprocess the
# image
if args["preprocess"] == "thresh":
	gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# make a check to see if median blurring should be done to remove
# noise
elif args["preprocess"] == "blur":
	gray = cv2.medianBlur(gray, 3)

# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file

#print("pytesseract.tesseract_cmd=", pytesseract.tesseract_cmd)

#lang ... 
text = pytesseract.image_to_string(Image.open(filename), lang=lang) #"BUL")
os.remove(filename)
print(text)
open("out.txt", "wb").write(text.encode("utf-8"))

# show the output images
# cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
