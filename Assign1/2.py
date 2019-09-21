import numpy as np
import matplotlib.pyplot as plt
import cv2

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
	print(x.shape, y.shape)
	for i in range(w):
		for j in range(h):
			reg = x[i:i+w1,j:j+w1]
			Conv_out[i,j] = np.sum(y*reg)

	return Conv_out

#Importing image data. Taking a grayscale image for computational ease

img  = cv2.imread('clown.tif',0)
# Predefining a 3*3 gaussian kernel
kern = np.array([[1,2,1],[2,4,2],[1,2,1]])/16.0
# Converting all the arrays to float to prevent int based artefacts (clipping)
img = img.astype(float)
kern = kern.astype(float)

out = Conv2D(img, kern)

plt.imshow(out, cmap='gray')
plt.show()