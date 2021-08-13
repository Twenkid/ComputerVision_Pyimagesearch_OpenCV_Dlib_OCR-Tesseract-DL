# Pose estimation using YOLO, ImageAI and DNN
# Authors:
# Tutorial by: https://learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/
# Model file links collection (replace .sh script): Twenkid
# http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel
#https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt
# ImageAI: https://github.com/OlafenwaMoses/ImageAI
# # YOLOv3:
# yolo.h5
# https://github-releases.githubusercontent.com/125932201/1b8496e8-86fc-11e8-895f-fefe61ebb499?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210813%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210813T002422Z&X-Amz-Expires=300&X-Amz-Signature=02e6839be131d27b142baf50449d021339cbb334eed67a114ff9b960b8beb987&X-Amz-SignedHeaders=host&actor_id=23367640&key_id=0&repo_id=125932201&response-content-disposition=attachment%3B%20filename%3Dyolo.h5&response-content-type=application%2Foctet-stream
# yolo-tiny.h5
# https://github-releases.githubusercontent.com/125932201/7cf559e6-86fa-11e8-81e8-1e959be261a8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20210812%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210812T232641Z&X-Amz-Expires=300&X-Amz-Signature=a5b91876c83b83a6aafba333c63c5f4a880bea9a937b30e52e92bbb0ac784018&X-Amz-SignedHeaders=host&actor_id=23367640&key_id=0&repo_id=125932201&response-content-disposition=attachment%3B%20filename%3Dyolo-tiny.h5&response-content-type=application%2Foctet-stream
# Todor Arnaudov - Twenkid: debug and merging, LearnOpenCV python code had a few misses, 13.8.2021
# It seems the pose model expects only one person so the image must be segmented first! pose1.jpg
# Detect with YOLO or ImageAI etc. then use DNN
# Specify the paths for the 2 files
# I tried with yolo-tiny, but the accuracy of the bounding boxes didn't seem acceptable.
#tf 1.15 for older versions of ImageAI - but tf doesn't support Py 3.8
#ImageAI: older versions require tf 1.x
#tf 2.4 - required by ImageAI 2.1.6 -- no GPU supported on Win 7, tf requires CUDA 11.0 (Win10). Win7: CUDA 10.x. CPU: works
# Set the paths to models, images etc.
# My experiments results: disappointingly bad pose estimation on the images I tested

import cv2
import tensorflow.compat.v1 as tf
from imageai.Detection import ObjectDetection
import os
boxes = []

def yolo():
    #name = "k.jpg"
    root = "Z:\\"
    name = "p1.jpg" #"2w.jpg" #"grigor.jpg" #"2w.jpg" #"pose1.webp" #1.jpg"
    execution_path = os.getcwd()
    yolo_path = "Z:\\yolo.h5"
    #yolo_path = "Z:\\yolo-tiny.h5"
    localdir = False

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    #detector.setModelTypeAsTinyYOLOv3()
    
    if localdir:
        detector.setModelPath(os.path.join(execution_path , yolo_path))
    else: 
        detector.setModelPath(yolo_path)

    #dir(detector)
    detector.loadModel()
    #loaded_model = tf.keras.models.load_model("./src/mood-saved-models/"model + ".h5")
    #loaded_model = tf.keras.models.load_model(detector.)

    #path = "E:\capture_023_29092020_150305.jpg" #IMG_20200528_044908.jpg"
    #pathOut = "E:\YOLO_capture_023_29092020_150305.jpg"

    #path = "pose1.webp" #E:\\capture_046_29092020_150628.jpg"
    pathOut = "yolo_out_2.jpg"


    
    path =  root + name
    pathOut = root + name + "yolo_out" + ".jpg"

    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , path), output_image_path=os.path.join(execution_path , pathOut), minimum_percentage_probability=10) #30)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")
    return detections, path

det,path = yolo()
yoloImage = cv2.imread(path) #crop regions from it 
for i in det:
  print(i)
  

protoFile = "Z:\\pose\\mpi\\pose_deploy_linevec_faster_4_stages.prototxt"
#protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
#weightsFile = "Z:\\pose\\mpi\\pose_iter_440000.caffemodel"
weightsFile = "Z:\\pose\\mpi\\pose_iter_160000.caffemodel"
#weightsFile = "pose_iter_160000.caffemodel"
#weightsFile = "pose_iter_440000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

"""
{'name': 'person', 'percentage_probability': 99.86668229103088, 'box_points': [1
8, 38, 153, 397]}
{'name': 'person', 'percentage_probability': 53.89136075973511, 'box_points': [3
86, 93, 428, 171]}
{'name': 'person', 'percentage_probability': 11.339860409498215, 'box_points': [
585, 99, 641, 180]}
{'name': 'person', 'percentage_probability': 10.276197642087936, 'box_points': [
126, 178, 164, 290]}
{'name': 'person', 'percentage_probability': 99.94878768920898, 'box_points': [2
93, 80, 394, 410]}
{'name': 'person', 'percentage_probability': 99.95986223220825, 'box_points': [4
78, 88, 589, 410]}
{'name': 'person', 'percentage_probability': 67.95878410339355, 'box_points': [1
, 212, 39, 300]}
{'name': 'person', 'percentage_probability': 63.609880208969116, 'box_points': [
153, 193, 192, 306]}
{'name': 'person', 'percentage_probability': 23.985233902931213, 'box_points': [
226, 198, 265, 308]}
{'name': 'sports ball', 'percentage_probability': 20.820775628089905, 'box_point
s': [229, 50, 269, 94]}
{'name': 'person', 'percentage_probability': 40.28712213039398, 'box_points': [4
23, 110, 457, 160]}
H, W, Ch 407 211 3
"""
yolo_thr = 70 #in percents, not 0.7
collected = []
bWiden = False
for d in det:
  if (d['name'] == 'person') and d['percentage_probability'] > yolo_thr:
    x1,y1,x2,y2 = d['box_points']
    if bWiden:
      x1-=20
      x2+=20
      y1-=30
      y2+=30
    cropped = yoloImage[y1:y2, x1:x2]    
    cv2.imshow(d['name']+str(x1), cropped)
    collected.append(cropped) #or copy first?
    cv2.waitKey()
    #x1,y1, ...

# for i in collected: cv2.imshow("COLLECTED?", i); cv2.waitKey()  #OK
    
# Read image
#frame = cv2.imread("Z:\\23367640.png") #1.jpg")
#src = "Z:\\2w.jpg" #z:\\pose1.webp" #nacep1.jpg"
#src = "z:\\pose1.webp" 
srcs = ["z:\\pose1.webp","Z:\\2w.jpg", "Z:\\grigor.jpg"]
id = 2
#src = srcs[2] 
src = path  #from first yolo, in order to compare

frame = cv2.imread(src)
cv2.imshow("FRAME"+src, frame)
#frameWidth, frameHeight, _ = frame.shape
frameHeight, frameWidth, ch = frame.shape
print("H, W, Ch", frameHeight, frameWidth, ch)
 
# Specify the input image dimensions
inWidth = 368 #184 #368
inHeight = 368 #184 #368

# Prepare the frame to be fed to the network
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

#cv2.imshow("G", inpBlob) #unsupported
#cv2.waitKey(0)

# Set the prepared object as the input blob of the network
net.setInput(inpBlob)
print(inpBlob)
output = net.forward()

print(output)

print("========")

H = output.shape[2]
W = output.shape[3]
# Empty list to store the detected keypoints
points = []
threshold = 0.3
maxKeypoints = 44
Keypoints = output.shape[1]
print("Keypoints from output?", Keypoints)
Keypoints = 15 #MPI ... returns only 15

labels = ["Head", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip", "Right Knee", "Right Ankle", "Left Hip", "Left Knee", "Left Ankle", "Chest", "Background"]

#for i in range(len()):
for i in range(Keypoints): #?
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold :
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        print(i, labels[i])
        print(x, y)
        points.append((int(x), int(y)))
    else :
        points.append(None)

print(points)

cv2.imshow("Output-Keypoints",frame)

def Detect(image):   #inWidth, Height ... - global, set as params later   
    frameHeight, frameWidth, ch = image.shape
    # Prepare the image to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    #cv2.imshow("G", inpBlob) #unsupported
    #cv2.waitKey(0)

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)
    print(inpBlob)
    output = net.forward()

    print(output)

    print("========")

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    threshold = 0.1
    maxKeypoints = 44
    Keypoints = output.shape[1]
    print("Keypoints from output?", Keypoints)
    Keypoints = 15 #MPI ... returns only 15

    labels = ["Head", "Neck", "Right Shoulder", "Right Elbow", "Right Wrist", "Left Shoulder", "Left Elbow", "Left Wrist", "Right Hip", "Right Knee", "Right Ankle", "Left Hip", "Left Knee", "Left Ankle", "Chest", "Background"]

    #for i in range(len()):
    for i in range(Keypoints): #?
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold :
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            print(i, labels[i])
            print(x, y)
            points.append((int(x), int(y)))
        else :
            points.append(None)

    print(points)
    cv2.imshow("Output-Keypoints",image)
    cv2.waitKey()

for i in collected: Detect(i)

cv2.waitKey(0)
cv2.destroyAllWindows()


