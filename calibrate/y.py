import numpy as np
import cv2

#yaml_data = np.asarray(cv2.Load("calibration.yaml"))

fs = cv2.FileStorage("calibration.yml", cv2.FILE_STORAGE_READ)
fn = fs.getNode("camera_matrix")
print(fn.mat())
dc = fs.getNode("dist_coeff")
print(dc.mat())

#print(yaml_data)


