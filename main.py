
# These are the libraries imported
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import time

import cv2

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#This is the main function to calculate optical flow using Lucas Kanade method
def opticalFlow(img1,img2,frameIndex,totalFrames): #It takes two images as inputs with the frameindex and total frames
	h,w = img1.shape[:2]
	colorImage1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
	#Here for pre-processing we are converting the images to greyscale
	img1G = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	img2G = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

	img1 = np.array(img1G)
	img2 = np.array(img2G)
	#Then we are performing guassian smoothing on it
	img1_smooth = cv2.GaussianBlur(img1,(3,3),0)
	img2_smooth = cv2.GaussianBlur(img2,(3,3),0)
	#Ix, Iy and It are the spatial derivatives being calculated by convolution
	Ix = signal.convolve2d(img1_smooth,[[-0.25, 0.25],[-0.25, 0.25]],'same') + signal.convolve2d(img2_smooth,[[-0.25, 0.25],[-0.25, 0.25]],'same')
	Iy = signal.convolve2d(img1_smooth,[[-0.25,-0.25],[ 0.25, 0.25]],'same') + signal.convolve2d(img2_smooth,[[-0.25,-0.25],[ 0.25, 0.25]],'same')
	It = signal.convolve2d(img1_smooth,[[ 0.25, 0.25],[ 0.25, 0.25]],'same') + signal.convolve2d(img2_smooth,[[-0.25,-0.25],[-0.25,-0.25]],'same')
	#Then we are finding the features of the smoothed image with 10000 maximum corners which we 
	# use afterwards in the lucas kanade algorithm
	features = cv2.goodFeaturesToTrack(img1_smooth,10000,0.01,10)	
	feature = np.int0(features)
	

	u = np.nan*np.ones((h,w))
	v = np.nan*np.ones((h,w))
	
	for l in feature:
		j,i = l.ravel()
		
		IX,IY,IT = [],[],[]
		
		if(i+2 < h and i-2 > 0 and j+2 < w and j-2 > 0):
			for b1 in range(-2,3):
				for b2 in range(-2,3):
					IX.append(Ix[i+b1,j+b2])
					IY.append(Iy[i+b1,j+b2])
					IT.append(It[i+b1,j+b2])
					
			LK = (IX,IY)
			LK = np.matrix(LK)
			LK_T = np.array(np.matrix(LK))
			LK = np.array(np.matrix.transpose(LK)) 
			
			A1 = np.dot(LK_T,LK)
			A2 = np.linalg.pinv(A1)
			A3 = np.dot(A2,LK_T)
			
			(u[i,j],v[i,j]) = np.dot(A3,IT)
	
	fig = plt.figure('')
	plt.subplot(1,1,1)
	plt.axis('off')
	plt.imshow(colorImage1, cmap = 'gray')
	#The if statement checks if the magnitude of either the horizontal (u) or vertical (v) component of the optical
	# flow vector at the current pixel exceeds the threshold t.
	#abs(u[i, j]) > t checks if the magnitude of the horizontal component at pixel (i, j) is greater than the threshold t.
	#abs(v[i, j]) > t checks if the magnitude of the vertical component at pixel (i, j) is greater than the threshold t.
	#If either of these conditions is true, it means that the optical flow at this pixel is significant enough to be visualized.
	#The plt.arrow function is then used to draw an arrow representing the optical flow vector.
	for i in range(h):
		for j in range(w):
			if abs(u[i,j]) > t or abs(v[i,j]) > t:
				plt.arrow(j,i,1.5*(-1*u[i,j]),1.5*(-1*v[i,j]), head_width = 3, head_length = 3, color = 'red')
	
	print('\r({:4}/{:4}) - Time Elapsed: {:10.10} seconds'.format(frameIndex+1,totalFrames,time.time()-start), end='')
	#Here the figure is again converted back to rgb
	fig.canvas.draw()
	img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
	plt.close()

	return img

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Here we are taking the video as input and calculating the total frames and fps of the video
start = time.time()
t = 0.7
video = cv2.VideoCapture('v.mp4')
totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(video.get(cv2.CAP_PROP_FPS))

_,image1 = video.read()
ret,image2 = video.read()
imageCounter = 0
#this part is for setting the name and format of the output file
videoName = 'v10_out.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
first = True
#Here as long as there are frames in the video we are passing 2 frames to the opticalflow function
#and adding the returned frame to the output video
while(ret):
	image = opticalFlow(image1,image2,imageCounter,totalFrames)
	
	if(not imageCounter):
		height,width = image.shape[:2]
		videoOut = cv2.VideoWriter(videoName, fourcc, fps, (width,height))

	videoOut.write(image)
	cv2.imshow("Optical Flow",image)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	
	image1 = image2.copy()
	ret,image2 = video.read()
	imageCounter += 1

print("")
video.release()
videoOut.release()
