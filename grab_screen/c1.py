#too slow! 0.22 - 0.25 sec?!.? for 640x480
#0.31 - 0.33 sec for 1600x1200
#-- include('examples/showgrabbox.py')--#
import pyscreenshot as ImageGrab
import sys

import time

if __name__ == '__main__':
    # part of the screen
    input = input("press enter...")
    st = time.time()
    im = ImageGrab.grab(bbox=(0, 0, 1600, 1200))
   # im = ImageGrab.grab(bbox=(10, 10, 510, 510))
    print("Elapsed time: ", time.time() - st)
    im.show()
    print("After show time: ", time.time() - st)
	
#-#

