from SimpleCV import Image, Color, Display
# load an image from imgur
img = Image('http://i.imgur.com/lfAeZ4n.png')
# use a keypoint detector to find areas of interest
feats = img.findKeypoints()
# draw the list of keypoints
feats.draw(color=Color.RED)
# show the  resulting image. 
img.show()
# apply the stuff we found to the image.
output = img.applyLayers()
# save the results.
output.save('juniperfeats.png')