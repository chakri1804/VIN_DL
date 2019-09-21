import numpy as np
import matplotlib.pyplot as plt
import cv2

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

img1 = cv2.imread('clown.tif',0)
# Converting all the arrays to float to prevent int based artefacts (clipping)
img1 = img1.astype(float)

# Predefining unsharpen kernel

# Most of the GIMP effects are brought to life by using either 5*5 or 3*3 kernels
unsh_kern = np.array([[0,0,0,0,0],[0,-2,-1,0,0],[0,-1,1,1,0],[0,0,1,2,0],[0,0,0,0,0]])
# Which is basically emboss kernel
unsh_out = Conv2D(img1, unsh_kern)

cv2.imwrite('embo_conv.png', unsh_out)

plt.imshow(unsh_out, cmap='gray')
plt.show()