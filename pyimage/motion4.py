#motion.py
import argparse
import datetime
import pyimagesearch.imutils as imutils
import time
import cv2

## Tried, corrected for OpenCV 4.x (findContours syntax) etc. by Todor Arnadov
#https://stackoverflow.com/questions/39475125/compatibility-issue-with-contourarea-in-opencv-3/39475245 -- hier,cnts ... opencv 4.1 ...
# The original tutorial is by Adrian Rosebrock
# https://www.pyimagesearch.com
# https://github.com/jrosebr1 
#
# Tracks differences between frames and marks them by contour-finding routines.
# Area parameter allows to ignore too small changes like noise.
# Purpose: to checks if there's someone (any activity) in the place, may be used
# to start/stop recording by a security or video-log camera.
#
# Sample usage:
# >python motion.py -v C:\Video\Home.mp4 -a 200

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)
 
# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None
# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	text = "Unoccupied"
	if (grabbed): cv2.imshow("Cap", frame)
	else: print("Error capturing frame")
 
	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break
 
	# resize the frame, convert it to grayscale, and blur it
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)	
	firstFrame = gray.copy()
	fd = 50#25
	thresh = cv2.threshold(frameDelta, fd, 255, cv2.THRESH_BINARY)[1]
 
	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	#py3, ... for py2: (cnts, _)
	thresh = cv2.dilate(thresh, None, iterations=2) 
    #cv 2
	#(_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
	#	cv2.CHAIN_APPROX_SIMPLE)
    #cv 4.x
	(cnts, hier) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break
	try:
		if cnts == None:
			continue
	except:
		pass       
	#if len(cnts)==0:    continue
	# loop over the contours
    
	for c in cnts:
		#print(c)
		#print(args["min_area"])
		#c = [[1.0 -1.0 -1.0 -1.0]]
		# if the contour is too small, ignore it
		#if len(c) == 0:	continue
		if cv2.contourArea(c) < args["min_area"]:
			continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		text = "Occupied"
    # draw the text and timestamp on the frame
	cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# show the frame and record if the user presses a key
	cv2.imshow("Security Feed", frame)
	cv2.imshow("Thresh", thresh)
	cv2.imshow("Frame Delta", frameDelta)
	#key = cv2.waitKey(1) & 0xFF
 
	## if the `q` key is pressed, break from the lop
	#if key == ord("q"):
	#	break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()