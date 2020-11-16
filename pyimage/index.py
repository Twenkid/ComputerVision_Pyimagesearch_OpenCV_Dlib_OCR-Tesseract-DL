# import the necessary packages
from pyimagesearch import RGBHistogram
import argparse
#import 
import _pickle as cPickle
import glob
import cv2

#construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required = True, help = "Path to where the computed index will be stored")
args = vars(ap.parse_args())
# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}# use glob to grab the image paths and loop over them
# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram.RGBHistogram([8, 8, 8])

for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
	# extract our unique image ID (i.e. the filename)
	k = imagePath[imagePath.rfind("/") + 1:]
	# load the image, describe it using our RGB histogram
	# descriptor, and update the index
	image = cv2.imread(imagePath)
	features = desc.describe(image)
	index[k] = features
#for imagePath in glob.glob(args["dataset"] + "/*.png"):
# extract our unique image ID (i.e. the filenam)

# we are now done indexing our image -- now we can write our
# index to disk
f = open(args["index"], "wb")
f.write(cPickle.dumps(index))
f.close()