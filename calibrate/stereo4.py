import numpy as np
import cv2
from matplotlib import pyplot as plt
# Author (experimenter): Todor Arnaudiv - Twenkid
# Combining and debugging examples, testing with histogram equalization and CLAHE
# Stereo2-3-4: 24.8.2021
# Histogram equalization, CLAHE, ... to compensate - can't adjust both cameras, one of them is always automatic?
# Logitech C270
# Bad results initially, but then I managed to find good parameters.

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
bWritePly = False #WARNING! Big files 1.7 MB per frame
uniquenessRatio = 1 #10

def SaveMtx(mtx, dist, path='calibration.yml'):
    #  Python code to write the image (OpenCV 3.2)
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', mtx)
    fs.write('dist_coeff', dist)
    fs.release()

#Twenkid: or 
#Load YAML camera calibration
#https://answers.opencv.org/question/31207/how-do-i-load-an-opencv-generated-yaml-file-in-python/
def loadyaml(path):
  import numpy as np
  import cv2  
  #fs = cv2.FileStorage("calibration.yml", cv2.FILE_STORAGE_READ)
  fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
  fn = fs.getNode("camera_matrix")
  print(fn.mat())
  dc = fs.getNode("dist_coeff")
  print(dc.mat())
  return fn, dc

calibrationFile = 'calibration.yml'
mtx, dist = loadyaml('calibration.yml')
print(dir(mtx))
#mtx = np.asarray(mtx)
#dist = np.asarray(dist)#.tolist()}
print(type(mtx))
print(type(dist))
print(dist)
print("========")
#print(type(mtx.mat()))
#print(type(dist.mat()))
w, h, ch = 640, 480, 1 #web camera default
#h,w,ch = imgL.shape

#np.array([[4,7,6],[1,2,5],[9,3,8]])
mtx = np.array([[454.32992171, 0., 220.08698296],[  0., 447.75773581, 275.09106524],[0.,0., 1.]])
dist =  np.array([-0.5151964, 0.55170682, -0.00955219,  0.05709782, -0.27482861])

"""
mtxL =[[978.89501953   0.         358.51610811]
 [  0.         891.51373291 381.00092072]
 [  0.           0.           1.        ]] (14, 32, 611, 435)
[[959.18787405   0.         330.16350476]
 [  0.         898.72749168 379.15124644]
 [  0.           0.           1.        ]]
[[ 0.90989593 -4.28831975  0.05118906  0.09287436  9.49483306]]
"""
#pink
mtxL = np.array([[1.31374475e+03, 0.00000000e+00,  3.14533534e+02],
 [0.00000000e+00,  1.11313623e+03,  2.63635793e+02],
 [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
#distL = [180, 181, 437, 168]
distL = np.array([-1.25805979e+00, -4.65137721e+01, -3.80789095e-03, -2.24134887e-01, 7.70739340e+02])

mtxR = np.array([[846.76654053, 0.,315.00730526],
 [  0.,         625.8104248,  243.6114669 ],
 [  0. ,          0.        ,   1.        ]]) #(219, 193, 182, 107)
distR = np.array([-6.26662115e-01, -1.22675535e+00 ,-1.24209685e-02,  5.29659784e-03, 4.62698811e+01])
"""
np.array([[276.4680481, 0.,256.53243226],
 [  0.,111.09597778,438.01447275],
 [  0. ,0. ,1. ]]) # (165, 410, 169, 61)
distR = np.array([-2.26268813,  3.79879042,   0.19441663,  -0.21170918,  -4.25948629])
"""
"""
[[4.59834961e+03 0.00000000e+00 3.00871285e+02]
 [0.00000000e+00 3.66075171e+03 2.17131118e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]] (190, 146, 222, 146)
[[-3.47604107e+01  9.74095928e+03 -1.13768426e-01 -1.60342524e-01
   3.94044307e+04]]
   
mtxR = np.array=([[4.59834961e+03, 0.00000000e+00,  3.00871285e+02], 
 [0.00000000e+00,  3.66075171e+03,  2.17131118e+02], 
 [0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
distR = np.array([-3.47604107e+01,  9.74095928e+03 ,-1.13768426e-01, -1.60342524e-01
   3.94044307e+04])
   """
"""
[[1.25101762e+03 0.00000000e+00 4.05277562e+02]
 [0.00000000e+00 1.28558338e+03 2.67407730e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
[[-1.25805979e+00 -4.65137721e+01 -3.80789095e-03 -2.24134887e-01
   7.70739340e+02]]
"""


#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx.mat(), dist.mat(), (w,h), 1, (w,h))
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
print(newcameramtx)

mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

bCam = False
bFirst = True
bCam = True
if bFirst and not bCam:
    capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.waitKey(2000)
    ret, imgL = capL.read()
    cv2.imwrite("left.jpg", imgL);    
    capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    print("Press key...")
    cv2.waitKey(2000)
    ret, imgR = capR.read()
    cv2.imwrite("right.jpg", imgL);
    print("Check the files...")
    capL.release()
    capR.release()  


imgL = cv2.imread('left.jpg',0)
imgR = cv2.imread('right.jpg',0)

print("Press ESC for exit...")
#stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
Eq = 2 #CLAHE, 1 = equ
bUndistort = True #False
#Eq = 0
window_size = 3 #3
speckle = 100
n = 0
if bCam:
  capL = cv2.VideoCapture(0, cv2.CAP_DSHOW)
  cv2.waitKey(1000)
  capR = cv2.VideoCapture(1, cv2.CAP_DSHOW)
  cv2.waitKey(1000)
  while True:
      ret,imgL = capL.read()
      print("Press a key...")
      k = cv2.waitKey(1)
      ret,imgR = capR.read()
      if not ret:
        k = cv2.waitKey(33)
        kk = k and 0xff
        if kk == 0x27: break
        continue
      imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
      imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
      #imgL = cv2.normalize(imgL, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      #imgR = cv2.normalize(imgR, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      
      #imgL = cv2.normalize(imgL, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      #imgR = cv2.normalize(imgR, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
      if Eq==1:
        imgL = cv2.equalizeHist(imgL)
        imgR = cv2.equalizeHist(imgR)
      elif Eq==2:
        clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
        imgL = clahe.apply(imgL)
        imgR = clahe.apply(imgR)
        
      mtx, dist
          
      #imgL.convertTo(imgL, cv2.CV_8UC1)
      #imgR.convertTo(imgR, cv2.CV_8UC1)
      cv2.imshow("L", imgL)
      cv2.imshow("R", imgR)
      k = cv2.waitKey(1)
      
      #"""
      disparity = stereo.compute(imgL,imgR)
      #cv2.imshow("DISPARITY", disparity) #gray
      norm_image = cv2.normalize(disparity, None, alpha = 0, beta = 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      cv2.imshow("DISPARITY_NORM", norm_image) #gray
      #plt.imshow(disparity,'gray') #OK
      #plt.show()
      
      if bUndistort:
          # undistort L
          dstL = cv2.undistort(imgL, mtxL, dist, None, newcameramtx)
          # crop the image
          x, y, w1, h1 = roi
          dstL = dstL[y:y+h1, x:x+w1]
          #cv.imwrite('calibresult_new.jpg', dst)

          # undistort
          mapx, mapy = cv2.initUndistortRectifyMap(mtxL, distL, None, newcameramtx, (w1,h1), 5)
          dstL = cv2.remap(imgL, mapx, mapy, cv2.INTER_LINEAR)
          # crop the image
          x, y, w1, h1 = roi
          dstL = dstL[y:y+h1, x:x+w1]
          
          ##########
          # undistort R
          dstR = cv2.undistort(imgR, mtxR, distR, None, newcameramtx)
          # crop the image
          x, y, w1, h1 = roi
          dstR = dstR[y:y+h1, x:x+w1]
          #cv.imwrite('calibresult_new.jpg', dst)

          # undistort
          mapx, mapy = cv2.initUndistortRectifyMap(mtxR, distR, None, newcameramtx, (w1,h1), 5)
          dstR = cv2.remap(imgL, mapx, mapy, cv2.INTER_LINEAR)
          # crop the image
          x, y, w1, h1 = roi
          dstR = dstR[y:y+h1, x:x+w1]     
          imgL = dstL
          imgR = dstR
      ##"""
      # disparity range is tuned for 'aloe' image pair
      #window_size = 3 #3
      min_disp = 16
      num_disp = 112-min_disp
      stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
      numDisparities = num_disp,
      blockSize = 16, #16,
      P1 = 8*3*window_size**2,
      P2 = 32*3*window_size**2,
      disp12MaxDiff = 1,
      uniquenessRatio = uniquenessRatio, #10,
      speckleWindowSize = 100, #100
      speckleRange = speckle, #32
      )

      print('computing disparity...')
      disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

      print('generating 3d point cloud...',)
      h, w = imgL.shape[:2]
      f = 1.17*w                         
      # C270: 1480 px of 1280 w... https://horus.readthedocs.io/en/release-0.2/source/scanner-components/camera.html guess for focal length
      #f = 1.5*w
      Q = np.float32([[1, 0, 0, -0.5*w],
                      [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                      [0, 0, 0,     -f], # so that y-axis looks up
                      [0, 0, 1,      0]])
      points = cv2.reprojectImageTo3D(disp, Q)
      colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
      mask = disp > disp.min()
      out_points = points[mask]
      out_colors = colors[mask]
      out_fn = 'out' + str(n) + '.ply'
      if bWritePly:
        write_ply(out_fn, out_points, out_colors)        
        print('%s saved' % out_fn)
      n+=1

      #cv.imshow('left', imgL)
      #cv.imshow('right', imgR)
      cv2.imshow('NEWdisparity', (disp-min_disp)/num_disp)
      k = cv2.waitKey(33)
      kk = k & 0xff
      if kk == 0x27: break
      if kk == ord('2'): window_size+= 1
      if kk == ord('1'): window_size-= 1
      if kk == ord('w'): speckle+= 10
      if kk == ord('q'): speckle-= 10
      if kk == ord('a'): uniquenessRatio+= 1
      if kk == ord('s'): uniquenessRatio-= 1
      if kk == ord('d'): bUndistort = not bUndistort
      print(window_size, speckle, uniquenessRatio, bUndistort, kk)
      
      
  
"""
#num = 10
#found = 0
#w = 0; h = 0
#w, h, ch = ...
"""

if bCam:
  ret, imgL = capL.read()
  print(ret, "imgL")
  ret, imgR = capR.read()
  print(ret, "imgR")

cv2.imshow("L", imgL)
cv2.imshow("R", imgR)
cv2.waitKey(0)
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
cv2.imshow("L", imgL)
cv2.imshow("R", imgR)
cv2.waitKey(0)
capL.release()
capR.release()  


#while(found < num):  # Here, 10 can be changed to whatever number you like to choose  
#    ret, img = cap.read() # Capture frame-by-frame
#    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
#    h,w,ch = img.shape    
    
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()