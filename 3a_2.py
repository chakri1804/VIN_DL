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
# predefining kernel    
Lapl = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

img = img.astype(float)
Lapl = Lapl.astype(float)

outL = Conv2D(img, Lapl)

plt.imshow(outL, cmap='gray')
plt.title('Laplacian edge detecting filter')
plt.show()

##### Scipy Implementation

outX = convolve2d(img, Lapl, 'same')
plt.imshow(outX, cmap='gray')
plt.title('Laplacian edge detecting filter - SCIPY')
plt.show()