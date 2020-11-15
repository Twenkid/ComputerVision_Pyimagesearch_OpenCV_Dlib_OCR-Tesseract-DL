# import the necessary packages
import numpy as np
import urllib.request
import cv2
 
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url) #urlopen
	print(resp)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image


# METHOD #2: scikit-image
from skimage import io

def showimages(urls): #list
	# loop over the image URLs
	for url in urls:
		# download the image using scikit-image
		print("downloading %s" % (url))
		image = io.imread(url)
		#cv2.imshow("Incorrect", image)
		cv2.imshow("Correct", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		cv2.waitKey(0)
        
im = url_to_image("http://razumir.twenkid.com/razumir-kakvomutryabva-korica-1.jpg")	
cv2.imshow("ANDR", im);
cv2.waitKey(0);

im = showimages( ["http://razumir.twenkid.com/img/3/pax_democratius_17_rzm.jpg", "http://razumir.twenkid.com/img/razumir2_1.png"])
