# Merging Youtube streams with pafy, opencv and multithreading
# Solution for a question by https://stackoverflow.com/users/10501459/fraser-langton
#https://stackoverflow.com/questions/65417658/use-multiprocessing-to-read-multiple-video-streams/65418212#65418212 
# Base code by Fraser Langton
# Refactored and debugged by Todor "Twenkid" Arnaudov: https://stackoverflow.com/users/6249888/twenkid
# Version: 23.12.2020 (Experimental code - some experiments left for educational purpose)
# https://github.com/Twenkid/Twenkid-FX-Studio/new/master/Py/y4.py

#3-4-2024: Pafy doesn't work anymore, but modified for file streams. Can be done for cameras as well, with indices of the cameras 0,1,2,3 VideoCapture(0) etc.
"""
Pafy in this implementation is not working with Youtube anymore , but the code can be used for file streams or camera streams (by replacing the open(...) with indices of the cameras 0,1, ...  The difference). For files:
def main():
    urls = [r"C:\v1.mp4, r"C:\v2.mp4", r"C:\v3.mp4", r"C:\v4.mp4" ]
     ...
    videos = [cv2.VideoCapture() for urls in urls]; 
    ...
   [video.open(best, apiPreference=cv2.CAP_ANY, params=None) for video, best in zip(videos, urls)]
  
The number of streams should be a power of two: 4, 9, 16, ... for correct merging.
"""

import multiprocessing #Process, Lock
from multiprocessing import Lock # Not needed
import cv2
import numpy as np
import pafy
import typing
import timeit
import time

urls = [
    "https://www.youtube.com/watch?v=tT0ob3cHPmE",
    "https://www.youtube.com/watch?v=XmjKODQYYfg",
    "https://www.youtube.com/watch?v=E2zrqzvtWio",

    "https://www.youtube.com/watch?v=6cQLNXELdtw",
    "https://www.youtube.com/watch?v=s_rmsH0wQ3g",
    "https://www.youtube.com/watch?v=QfhpNe6pOqU",

    "https://www.youtube.com/watch?v=C_9x0P0ebNc",
    "https://www.youtube.com/watch?v=Ger6gU_9v9A",
    "https://www.youtube.com/watch?v=39dZ5WhDlLE"
]

#Todor: Merging seems to require equal number of sides, so 2x2, 3x3 etc.
'''
[    
    "https://www.youtube.com/watch?v=C_9x0P0ebNc",
    "https://www.youtube.com/watch?v=Ger6gU_9v9A",
    "https://www.youtube.com/watch?v=39dZ5WhDlLE",   
    "https://www.youtube.com/watch?v=QfhpNe6pOqU",
]
'''
'''
    "https://www.youtube.com/watch?v=tT0ob3cHPmE",
    "https://www.youtube.com/watch?v=XmjKODQYYfg",
    "https://www.youtube.com/watch?v=E2zrqzvtWio"

    "https://www.youtube.com/watch?v=6cQLNXELdtw",
    "https://www.youtube.com/watch?v=s_rmsH0wQ3g",
    "https://www.youtube.com/watch?v=QfhpNe6pOqU",

    "https://www.youtube.com/watch?v=C_9x0P0ebNc",
    "https://www.youtube.com/watch?v=Ger6gU_9v9A",
    "https://www.youtube.com/watch?v=39dZ5WhDlLE"
'''    

width = np.math.ceil(np.sqrt(len(urls)))
dim = 1920, 1080

streams = []
bestStreams = []

def main():
    global bestStreams
    urls = [r"C:\Canon\A540\2-4-2024\MVI_8400.AVI", r"C:\Canon\A540\2-4-2024\MVI_8401.AVI", r"C:\Canon\A540\2-4-2024\MVI_8446.AVI", r"C:\Canon\A540\2-4-2024\MVI_8447.AVI" ]
    #[pafy.new(url).getbest() for url in urls]
    print(streams)
    [bestStreams for best in streams]
    print(bestStreams)
    cv2.waitKey(0)
    v1 = cv2.VideoCapture(urls[0], apiPreference=cv2.CAP_ANY, params = None)
    v2 = cv2.VideoCapture(urls[1], apiPreference=cv2.CAP_ANY, params = None)
    v3=cv2.VideoCapture(urls[2], apiPreference=cv2.CAP_ANY, params = None)
    v4=cv2.VideoCapture(urls[3], apiPreference=cv2.CAP_ANY, params = None)
    
    videos = [cv2.VideoCapture() for urls in urls]
    bestURLS = [] 
    
    #[video.open(best.url) for video, best in zip(videos, streams)]
    
    #[bestURLS.append(best.url) for best in streams]
    bestURLS = urls
    #"""
    [video.open(best, apiPreference=cv2.CAP_ANY, params=None) for video, best in zip(videos, urls)]
    
    #"""
    #videos = [v1, v2, v3] #, v4]
    
    
    #[ for video, best in zip(videos, streams)]
    print(videos) #bestURLS)
    cv2.waitKey(0)
    cv2.namedWindow('Video', cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    LOCK = Lock()
    #proc = get_framesUL(bestStreams, LOCK)
    #proc, pipes = get_framesULJ(bestStreams, LOCK)
    proc, pipes = get_framesULJ(bestURLS, LOCK)     
    print("PROC, PIPES", proc, pipes)
    #cv2.waitKey(0)
    frames = []
    while True:
        start_time = timeit.default_timer()
        # frames = [cv2.resize(video.read()[-1], (dim[0] // width, dim[1] // width)) for video in videos]
        #frames = get_frames(videos, LOCK)
        #frames = get_framesUL(streams, LOCK)
        
        
        print(timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        
        frames = [x.recv() for x in pipes]
        lf = len(frames)
        print("LEN(FRAMES)=", lf);
        #if lf<3: time.sleep(3); print("LEN(FRAMES)=", lf); #continue #Else merge and show
        #proc.join()
        #elif lf==3: frames = [x.recv() for x in pipes]
                
        dst = merge_frames(frames)
        print(timeit.default_timer() - start_time)
         
        start_time = timeit.default_timer()      
        #if cv2!=None:
        try:
          cv2.imshow('Video', dst)
        except: print("Skip")
        cv2.waitKey(1)  

        if cv2.waitKey(10) & 0xFF == ord('e'):
            break
        print(timeit.default_timer() - start_time)

        continue
        
    for proc in jobs:
        proc.join()
        
    [video.release() for video in videos]
    cv2.destroyAllWindows()



def get_framesULJ(videosURL, L): #return the processes, join in main and read the frames there
    # frames = [video.read()[-1] for video in videos]
    print("get_framesULJ:",videosURL)    
    jobs = []
    pipe_list = []
    #print("VIDEOS:",videosURL)    
    #for video in videos:
    for videoURL in videosURL: #urls:
        recv_end, send_end = multiprocessing.Pipe(False)
        print(recv_end, send_end)
        p = multiprocessing.Process(target=get_frame2L, args=(videoURL, send_end, L))
        #p = multiprocessing.Process(target=get_frame, args=(video, send_end, L))
        #if (p==None): continue
        print("P = ", p)
        #time.sleep(0.001)
        jobs.append(p)
        print("JOBS, len", jobs, len(jobs))                
        pipe_list.append(recv_end)
        print("pipe_list", pipe_list)               
        p.start()
        #cv2.waitKey(0)

    #for proc in jobs:
    #    proc.join()

    #frames = [x.recv() for x in pipe_list]
    #return frames
    #cv2.waitKey(0)
    return jobs, pipe_list

def get_frame2L(videoURL, send_end, L):
    v = cv2.VideoCapture()
    #[video.open(best.url)
    #L.acquire()
    v.open(videoURL)
    print("get_frame2", videoURL, v, send_end)
    #cv2.waitKey(0)
    while True:      
      ret, frame = v.read()
      if ret: send_end.send(frame); #cv2.imshow("FRAME", frame); cv2.waitKey(1)   
      else: print("NOT READ!"); break
    #send_end.send(v.read()[1])
    #L.release()
    
def get_framesUL(videosURL, L):
    # frames = [video.read()[-1] for video in videos]

    jobs = []
    pipe_list = []
    print("VIDEOS:",videosURL)    
    #for video in videos:
    for videoURL in videosURL: #urls:
        recv_end, send_end = multiprocessing.Pipe(False)
        print(recv_end, send_end)
        p = multiprocessing.Process(target=get_frame2L, args=(videoURL, send_end, L))
        #p = multiprocessing.Process(target=get_frame, args=(video, send_end, L))
        #if (p==None): continue
        print("P = ", p)
        #time.sleep(0.001)
        jobs.append(p)
        print("JOBS, len", jobs, len(jobs))                
        pipe_list.append(recv_end)
        print("pipe_list", pipe_list)               
        p.start()

    for proc in jobs:
        proc.join()

    frames = [x.recv() for x in pipe_list]
    return frames


def get_frames(videos, L):
    # frames = [video.read()[-1] for video in videos]

    jobs = []
    pipe_list = []
    print("VIDEOS:",videos)    
    for video in videos:
        recv_end, send_end = multiprocessing.Pipe(False)
        print(recv_end, send_end)
        p = multiprocessing.Process(target=get_frame, args=(video, send_end, L))
        #p = multiprocessing.Process(target=get_frame, args=(video, send_end, L))
        #if (p==None): continue
        print("P = ", p)
        #time.sleep(0.001)
        jobs.append(p)
        print("JOBS, len", jobs, len(jobs))                
        pipe_list.append(recv_end)
        print("pipe_list", pipe_list)               
        p.start()

    for proc in jobs:
        proc.join()

    frames = [x.recv() for x in pipe_list]
    return frames
    
def get_frame(video, send_end, L):
    L.acquire()
    print("get_frame", video, send_end)
    send_end.send(video.read()[1])
    L.release()
    # send_end.send(cv2.resize(video.read()[1], (dim[0] // width, dim[1] // width)))

    
def get_frame2(videoURL, send_end):
    v = video.open(videoURL)       
    while True:
      ret, frame = v.read()
      if ret: send_end.send(frame)
      else: break
      
    
def merge_frames(frames: typing.List[np.ndarray]):
    #cv2.imshow("FRAME0", frames[0]) ########## not images/small
    #cv2.imshow("FRAME1", frames[1]) ##########
    #cv2.imshow("FRAME2", frames[2]) ##########
    #cv2.imshow("FRAME3", frames[3]) ##########
    #cv2.waitKey(1)
    width = np.math.ceil(np.sqrt(len(frames)))
    rows = []
    for row in range(width):
        i1, i2 = width * row, width * row + width
        rows.append(np.hstack(frames[i1: i2]))
    
    
    return np.vstack(rows)


if __name__ == '__main__':
    main()
