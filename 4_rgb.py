import numpy as np
import matplotlib.pyplot as plt
import cv2

# Assuming x to be the input image and y to be kernel
# Since it is convolution, we'd like to pad the image with zeros

def Conv2D(x,y):
	w,h,d = np.shape(x)
	# Since square kernel, ignoring one of the dimension values
	w1,_ = np.shape(y)
	# Predefining a 0 filled numpy array to store the convolution outputs
	Conv_out = np.zeros_like(x)
	# Flipping kernel
	y = np.fliplr(np.flipud(y))
	# Padding the input image
	pad_w = int((w1-1)/2)
	x = np.pad(x, ((pad_w,pad_w),(pad_w,pad_w),(0,0)), 'constant')
	print(x.shape, y.shape)
	for i in range(w):
		for j in range(h):
			for k in range(d):
				reg = x[i:i+w1,j:j+w1, k]
				Conv_out[i,j,k] = np.sum(y*reg)
	return Conv_out

def get_gauss_kernel(size=11,sigma=0.5):
    center=int(size/2)
    kernel=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            diff=np.sqrt((i-center)**2+(j-center)**2)
            kernel[i,j]=np.exp(-(diff**2)/(2*sigma**2))
    return kernel/np.sum(kernel)

def Hybrid_imgs(img1, img2, kern_size, sigm1, sigm2):
	kernel1 = get_gauss_kernel(kern_size, sigm1)
	kernel2 = get_gauss_kernel(kern_size, sigm2)
	img1_low_freq = Conv2D(img1, kernel1)
	img2_low_freq = Conv2D(img2, kernel2)
	img2_high_freq = img2 - img2_low_freq
	hybrid = (img1_low_freq + img2_high_freq)
	hybrid = hybrid/np.max(hybrid)
	return hybrid 

#Importing image data. Taking a grayscale image for computational ease
img1 = cv2.imread('data/dog.bmp')
img2 = cv2.imread('data/cat.bmp')
# Converting all the arrays to float to prevent int based artefacts (clipping)
img1 = img1.astype(float)
img2 = img2.astype(float)

hyb_out = Hybrid_imgs(img1, img2, 11, 10, 100)
# Flipping because OpenCV has BGR format instead of RGB
hyb_out = np.flip(hyb_out, axis=2)
plt.imshow(hyb_out)
plt.savefig('4_hybrid_rgb.png')
plt.show()