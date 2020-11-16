# import the necessary packages
from pyimagesearch import Search
import numpy as np
import argparse
import _pickle as cPickle
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory that contains the images we just indexed")
ap.add_argument("-i", "--index", required = True,
	help = "Path to where we stored our index")
args = vars(ap.parse_args())
# load the index and initialize our searcher
index = cPickle.loads(open(args["index"], "rb").read()) # "rb") #, encoding='utf-8'
searcher = Search.Searcher(index)
# loop over images in the index -- we will use each one as
# a query image
for (query, queryFeatures) in index.items():
	# perform the search using the current query
	results = searcher.search(queryFeatures)
	# load the query image and display it
	#path = args["dataset"] + "/%s" % (query)
	print()
	#path = args["dataset"] + "\\"+ query
	path = query
	print(path)
	queryImage = cv2.imread(path)
    
	cv2.imshow("Query", queryImage)
	print ("query: %s" % (query))
	# initialize the two montages to display our results --
	# we have a total of 25 images in the index, but let's only
	# display the top 10 results; 5 images per montage, with
	# images that are 400x166 pixels
	
	#don't use
	montageA = np.zeros((166 * 5, 400, 3), dtype = "uint8")
	montageB = np.zeros((166 * 5, 400, 3), dtype = "uint8")
	
	# loop over the top ten results
	for j in range(0, 10):  #xrange
		# grab the result (we are using row-major order) and
		# load the result image
		(score, imageName) = results[j]
		
		path = args["dataset"] + "/%s" % (imageName)
		#p = path.replace("\\\\", "#")
		#print(p)
		
		#p = p.replace("##", "\\")
		#print(p)
		#p = path.replace("#", "\\")
		#path = p
		print('===============')
		print(path)
		print('===============')
		split = path.split("/") ###################
		#path = imageName
		#result = cv2.imread(path)
		result = cv2.imread(split[1])
		cv2.imshow("Result " + str(j), result)
		cv2.waitKey(0)
		print ("\t%d. %s : %.3f" % (j + 1, imageName, score))
		# check to see if the first montage should be used
		#cv2.imshow("Result" + str(j), result)
		#if j < 5:
		#	montageA[j * 166:(j + 1) * 166, :] = result
		# otherwise, the second montage should be used
		#else:
		#	montageB[(j - 5) * 166:((j - 5) + 1) * 166, :] = result
	# show the results
	#cv2.imshow("Results 1-5", montageA)
	#cv2.imshow("Results 6-10", montageB)
	cv2.destroyAllWindows()
	cv2.waitKey(0)