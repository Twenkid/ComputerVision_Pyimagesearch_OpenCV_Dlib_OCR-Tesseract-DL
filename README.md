ComputerVision_OpenCV

Computer vision tests, Python OpenCV code, originally from Pyimagesearch and other tutorials and examples, sometimes aggregated or refactored or extended and various programming and API tests.

 The original Pyimagesearch tutorials by Adrian Rosebrock:  https://www.pyimagesearch.com  
 https://github.com/jrosebr1 

## OpenCV build tips by Todor

## Speed up build:

**Compile from/to RAM disks** (in Windows). **OSFMount, imdisk**, etc. Don't forget to format the new drives.

_(!!! Beware with imdisk and unmounting! Unmounting one drive may delete the others as well (at least I've had such an issue)._
Both RAM Drivers allow to extend the disks.

## Storage

OCV 4.5.2 could be > 1.5 GB after built, depends on the options and if you add opencv_contribute. (I don't know for rich builds with Gstreamer etc.).
Allocate ~ 2.5 GB for the built output drive.

```
OPENCV_EXTRA_MODULES_PATH  D:/opencv_contrib-master/opencv_contrib-master/modules
```

After downloading them from OpenCV github.
...

**CUDA_ARCH_BIN**
3.0;3.5;3.7;5.0;5.2;6.0;6.1;7.0;7.5

```
--> Change to **just the target one**, if only for your current PC, this is the slowest part of compilation.
https://answers.opencv.org/question/5090/why-opencv-building-is-so-slow-with-cuda/
```
750 Ti: 5.0;
960: 5.2 
1070: 6.1
2060 Super: 7.5.
```

CPU dispatch: if you don't plan to use old CPUs, set the highest minimum.

Core 2 Quad: SSE 4.1
i5 2500, 3220M, 3470: AVX
Skylake: AVX2

Don't forget:

```
WITH_OPEN_GL
CUDA ...
OPENMP
DSHOW
D3D11 ...
...
```

Install gstreamer if needed:  (slow download, requires 1 GB+ if all is installed, 1.18)
https://gstreamer.freedesktop.org/download/

Follow: https://galaktyk.medium.com/how-to-build-opencv-with-gstreamer-b11668fa09c
Windows: 
Set Path: (Start->Envir... ) “GSTREAMER_DIR” “C:\gstreamer\1.0\x86_64”
...

Install QT if you need it etc.

https://github.com/Twenkid/ComputerVision_Pyimagesearch_OpenCV_Dlib_OCR-Tesseract-DL/blob/master/opencv_build_tips.md

