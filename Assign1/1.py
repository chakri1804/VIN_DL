import numpy as np
import matplotlib.pyplot as plt
import cv2

# Normalised Cross Correlation requires both the template and the section of the image
# We're comparing with the template to be normalised

def normalise(x):
	temp = (x-np.mean(x))/np.std(x)
	return temp

# Assuming the stride for NCC to be 1
# x is the template
# y is the image on which we are trying to template match

def NCC(x,y):
	w,h,_ = np.shape(x)
	w1,h1,_ = np.shape(y)
	fin_w = w1-w+1
	fin_h = h1-h+1
	NCC_out = np.zeros((fin_w,fin_h))
	for i in range(fin_w):
		for j in range(fin_h):
			x = normalise(x)
			reg = normalise(y[i:i+w,j:j+h])
			NCC_out[i,j] = np.mean(x*reg)
	return NCC_out

#Importing image data. Note that template has been cropped from hollow.jpeg at pixel index (122,61)

# mask = cv2.imread('template.jpg', 0)
img  = cv2.imread('hollow.jpg')
mask = cv2.imread('mask.jpg')

print(np.shape(mask))
print(np.shape(img))



out = NCC(mask, img)

# print(out)
ind = np.unravel_index(np.argmax(out, axis=None), out.shape)
print(ind)
print(out[ind])