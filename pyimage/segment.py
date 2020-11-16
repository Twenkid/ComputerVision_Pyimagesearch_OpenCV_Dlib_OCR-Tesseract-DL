# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())
# load the image and apply SLIC and extract (approximately)
# the supplied number of segments
image = cv2.imread(args["image"])
print("Loaded image", args["image"])
print("segments = slic(img_as_float(image)... Working")
#segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
segments = slic(img_as_float(image), n_segments = 10, sigma = 5)
 
 
# show the output of SLIC
fig = plt.figure("Superpixels")
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
plt.axis("off")
plt.show()
# loop over the unique segment values
for (i, segVal) in enumerate(np.unique(segments)):
	# construct a mask for the segment
	print ("[x] inspecting segment %d" % (i))
	mask = np.zeros(image.shape[:2], dtype = "uint8")
	mask[segments == segVal] = 255
 
	# show the masked region
	cv2.imshow("Mask", mask)
	Masked = cv2.bitwise_and(image, image, mask = mask)
    #cv2.imshow("Applied", cv2.bitwise_and(image, image, mask = mask))
	cv2.imshow("Applied",Masked)
    #Second level segmentation #but also cut the image?
    #Scan, find bounding box etc.?
	seg2 = slic(img_as_float(Masked), n_segments = 5, sigma = 5)
	for (j,segV) in  enumerate(np.unique(seg2)): #no -- it should crop the selected region from the image 
		print ("[x] inspecting segment %d.%d" % (i,j))
		mask2 = np.zeros(image.shape[:2], dtype = "uint8")
		mask2[seg2 == segV] = 255
		Masked2 = cv2.bitwise_and(image, image, mask = mask2)
		cv2.imshow("Second Level",Masked2)
		cv2.waitKey(0)
 
	# show the masked region
	cv2.imshow("Mask", mask)     
	cv2.waitKey(0)