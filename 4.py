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

def get_gauss_kernel(size=3,sigma=1):
    center=int(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            diff=np.sqrt((i-center)**2+(j-center)**2)
            kernel[i,j]=np.exp(-(diff**2)/(2*sigma**2))
    return kernel/np.sum(kernel)

def Hybrid_imgs(img1, img2, kern_size, sigm):
	kernel = get_gauss_kernel(kern_size, sigm)
	img1_low_freq = Conv2D(img1, kernel)
	img2_low_freq = Conv2D(img2, kernel)
	img2_high_freq = img2 - img2_low_freq
	hybrid = img1_low_freq + img2_high_freq
	return hybrid 

#Importing image data. Taking a grayscale image for computational ease
img1 = cv2.imread('data/einstein.bmp',0)
img2 = cv2.imread('data/marilyn.bmp', 0)
# Converting all the arrays to float to prevent int based artefacts (clipping)
img1 = img1.astype(float)
img2 = img2.astype(float)

hyb_out = Hybrid_imgs(img1, img2, 5, 7)

plt.imshow(hyb_out, cmap='gray')
plt.show()