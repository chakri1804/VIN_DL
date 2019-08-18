#Sobel

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

# Assuming x to be the input image and y to be kernel
# Since it is convolution, we'd like to pad the image with zeros

def Conv2D(x,y):
	w,h = np.shape(x)
	# Since square kernel, ignoring one of the dimension values
	w1,_ = np.shape(y)
	# Predefining a 0 filled numpy array to store the convolution outputs
	Conv_out = np.zeros_like(x)
	# Flipping kernel
	y = np.fliplr(np.flipud(y))
	# Padding the input image
	pad_w = int((w1-1)/2)
	x = np.pad(x, pad_w, 'constant')
	# print(x.shape, y.shape)
	for i in range(w):
		for j in range(h):
			reg = x[i:i+w1,j:j+w1]
			Conv_out[i,j] = np.sum(y*reg)
	return Conv_out

img  = cv2.imread('clown.tif',0)

# Predefining two 3*3 Sobel kernels, one for horizontal and the other for vertical edge detections  
SobelX = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
SobelY = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

img = img.astype(float)
SobelX = SobelX.astype(float)
SobelY = SobelY.astype(float)

outX = Conv2D(img, SobelX)
outY = Conv2D(img, SobelY)

plt.imshow(outX, cmap='gray')
plt.title('Horizontal edge detecting Sobel filter')
plt.show()
plt.imshow(outY, cmap='gray')
plt.title('Vertical edge detecting Sobel filter')
plt.show()

# Approximate 

out_apprx = np.abs(outX) + np.abs(outY)
plt.imshow(out_apprx, cmap='gray')
plt.title('Approximate Sobel Edge detection')
plt.show()

# Root Mean squared

out_RMS = np.sqrt(outX**2 + outY**2)
plt.imshow(out_RMS, cmap='gray')
plt.title('RMS Sobel Edge detection')
plt.show() 

##### Scipy Implementation

outX = convolve2d(img, SobelX, 'same')
plt.imshow(outX, cmap='gray')
plt.title('Horizontal edge detecting Sobel filter - SCIPY')
plt.show()