#from skimage import io
from skimage import io #, data, filters #data, io, filters

#import scikit.image.io
#import scipy #.image.io
import cv2
 
# loop over the image URLs
#for url in urls:
# download the image using scikit-image
#print "downloading %s" % (url)
url = r"http://twenkid.com/img/saitut-se-risuva.gif"
image = io.imread(url)
cv2.imshow("Incorrect", image)
cv2.imshow("Correct", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

from skimage import data, filters

image = data.coins()
# ... or any other NumPy array!
edges = filters.sobel(image)
io.imshow(edges)
io.show()

cv2.waitKey(0)