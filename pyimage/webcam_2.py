# import the necessary packages
from threading import Thread
import cv2
import time #sleep

class WebcamVideoStream:

	def setsleep(self, sleep=0.01):
            self.sleep = sleep;
            
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		self.cnt = 0
		self.sleep = 0.01;

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, args=())
		t.daemon = False # True #False # True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		#if not self.stopped:
		print("Update")
		#if (self.cnt>50): return
		#while not self.stopped:
		while True:                
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				print("stop thread")
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()
			time.sleep(self.sleep) #0.01)
			self.cnt+=1
			#if (self.cnt > 50): print("Stop thread due a limit. Press 'q' to exit or whatever"); return
			#print(self.cnt)
			

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
