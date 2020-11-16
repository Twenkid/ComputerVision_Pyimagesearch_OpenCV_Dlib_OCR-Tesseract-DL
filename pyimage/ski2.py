import skimage
from skimage import transform
from skimage import filters
import argparse
import cv2
#https://github.com/scikit-image/scikit-image/issues/3646
#Seam carving is patented and removed ...

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image file")
ap.add_argument("-d", "--direction", type=str,
	default="vertical", help="seam removal direction")
args = vars(ap.parse_args())

# load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# compute the Sobel gradient magnitude representation
# of the image -- this will serve as our "energy map"
# input to the seam carving algorithm
mag = filters.sobel(gray.astype("float"))

cv2.imshow("Sobel", mag)

# show the original image
cv2.imshow("Original", image)
# loop over a number of seams to remove
for numSeams in range(20, 140, 20):
	# perform seam carving, removing the desired number
	# of frames from the image -- `vertical` cuts will
	# change the image width while `horizontal` cuts will
	# change the image height
	carved = transform.seam_carve(image, mag, args["direction"],numSeams)
	print("[INFO] removing {} seams; new size: "
		"w={}, h={}".format(numSeams, carved.shape[1],
			carved.shape[0]))
 
	# show the output of the seam carving algorithm
	cv2.imshow("Carved", carved)
	cv2.waitKey(0)