import numpy as np
import argparse
import cv2

# -i image
# c12.py
# Demonstrates a number of filters: Laplacian, Sobel, Hough Circles and sending
# a list of functions as a parameter for calling.
# Press ESC to exit.
#
# Based on the original tutorials and examples by Adrian Rosebrock 
# https://www.pyimagesearch.com
# https://github.com/jrosebr1 
# 
# Aggregated by Todor Arnaudov
# https://github.com/Twenkid
# https://artificial-mind.blogspot.bg

def Circles(image, gray): 
  # detect circles in the image  
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)
  output = image.copy()
 
  # ensure at least some circles were found
  if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
 
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        print(x,y,r)
    cv2.imshow("Circles", np.hstack([image, output]))
  else: print("No circles found!") 
  # show the output image
  
  #cv2.waitKey(0)

def Laplacian(image, gray):
  #lap = cv2.Laplacian(image, cv2.CV_64F)
  lap = cv2.Laplacian(gray, cv2.CV_32F)
  lap = np.uint8(np.absolute(lap))
  cv2.imshow("Laplacian", lap)
  
def Sobel(image, gray):
  sobelX = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
  sobelY = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
  sobelX = np.uint8(np.absolute(sobelX))
  sobelY = np.uint8(np.absolute(sobelY))

  sobelCombined = cv2.bitwise_or(sobelX, sobelY)
  cv2.imshow("Sobel X", sobelX)
  cv2.imshow("Sobel Y", sobelY)
  cv2.imshow("Sobel Combined", sobelCombined)

def All():
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required = True,
  help = "Path to the image")
  args = vars(ap.parse_args())

  image = cv2.imread(args["image"])
  #output = image.copy()

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  cv2.imshow("Original", image)
  
  Laplacian(image, gray)
  cv2.waitKey(0)
  
  Sobel(image, gray)
  cv2.waitKey(0)
  
  Circles(image, gray)
  
  cv2.waitKey(0)
 
def GetArguments():
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required = True,
  help = "Path to the image")
  args = vars(ap.parse_args())
  return args
  
def GetImages(args):
  image = cv2.imread(args["image"])
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  return image, gray
  
def ShowOriginal(image):
  cv2.imshow("Original", image)   

def CallFunctions(image, *argv): #Todor: convenient calling of many sample functions
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for f in argv:
    f(image, gray)
	
def CallFunctionsList(image, functions): #Todor: convenient calling of many sample functions
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  for f in functions:
    f(image, gray)
	
def Short(): #10-7-2017
  #args = GetArguments()
  #image, gray = GetImages(args)
  
  image, gray = GetImages(GetArguments())
  ShowOriginal(image)        
  
  #CallFunctionsList(image, [Laplacian, Sobel, Circles])    
  CallFunctions(image, Laplacian, Sobel, Circles)
  
def Wait():
  #print("Press a key...")
  cv2.waitKey(0)
  
Short()
Wait()
#All()