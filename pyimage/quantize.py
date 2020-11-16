# import the necessary packages
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
ap.add_argument("-c", "--clusters", required = True, type = int,
	help = "# of clusters")
args = vars(ap.parse_args())
# load the image and grab its width and height
image = cv2.imread(args["image"])
imageCopy = image.copy()
(h, w) = image.shape[:2]
 
# convert the image from the RGB color space to the L*a*b*
# color space -- since we will be clustering using k-means
# which is based on the euclidean distance, we'll use the
# L*a*b* color space where the euclidean distance implies
# perceptual meaning
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
 
# reshape the image into a feature vector so that k-means
# can be applied
image = image.reshape((image.shape[0] * image.shape[1], 3))
 
# apply k-means using the specified number of clusters and
# then create the quantized image based on the predictions

clustStart = args["clusters"] #+

clt = MiniBatchKMeans(n_clusters = args["clusters"])
labels = clt.fit_predict(image)
quant = clt.cluster_centers_.astype("uint8")[labels]
 
# reshape the feature vectors to images
quant = quant.reshape((h, w, 3))
image = image.reshape((h, w, 3))
 
# convert from L*a*b* to RGB
quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
 
# display the images and wait for a keypress
#cv2.imshow("image", np.hstack([image, quant]))  
#Todor Arnaudov: If the image is too wide, doesn't fit the screen - it's not displayed as two images,
#thus showing them one after another


images = []
images.append(quant)

cv2.imshow("image", quant)
cv2.waitKey(0)

clustTarget = clustStart + 24
clust = clustStart

print("Quantizing 6 more steps, press a key each time")

while(clust<clustTarget):
  image = imageCopy.copy()
  image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
  image = image.reshape((image.shape[0] * image.shape[1], 3)) #converts to h*w, 1 dimension
  clt = MiniBatchKMeans(n_clusters = clust) #args["clusters"])
  labels = clt.fit_predict(image)
  quant = clt.cluster_centers_.astype("uint8")[labels]
  # reshape the feature vectors to images
  quant = quant.reshape((h, w, 3))
  image = image.reshape((h, w, 3))
 
  # convert from L*a*b* to RGB
  quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
  image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)  
  #for i in range (2, 24):  
  images.append(quant)
  cv2.imshow("image", quant) 
  cv2.waitKey(0)
  clust+=1

while(True):
  for im in images:
     cv2.imshow("image", im)
     k = cv2.waitKey(0)
     #if (k & 0xFF =='q'): break
     if (k ==27): break
   
 
