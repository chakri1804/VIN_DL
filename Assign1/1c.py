import numpy as np
import matplotlib.pyplot as plt
import cv2

# Normalised Cross Correlation requires both the template and the section of the image
# We're comparing with the template to be normalised

# Adding 1e-6 to maintain numerical stability

def normalise(x):
	temp = (x-np.mean(x))/(np.std(x)+1e-6)
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

#Importing image data.
img  = cv2.imread('u2cuba.jpg')
mask = cv2.imread('trailerSlightlyBigger.png')

print(np.shape(mask))
print(np.shape(img))

out = NCC(mask, img)

# print(out)
ind = np.unravel_index(np.argmax(out, axis=None), out.shape)
print("The template is most similar to image starting at the pixel :")
print(ind)
print("Cross Correlation value :")
print(out[ind])
plt.imshow(out, cmap='gray')
plt.savefig('1c_artefacts.png')
plt.show()