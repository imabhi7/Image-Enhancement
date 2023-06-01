import cv2 as cv                                        #importing the open cv library for image processing
import numpy as np                                      #importing the numpy library as np to create kernel matrix
from matplotlib import pyplot as plt                    #importing the plotting library to plot histogram

file_location = input('Enter image location : ')

img = cv.imread(file_location)           #reading image using imread function in cv2
img = cv.resize(img, (640,480))                         #resizing the image so that it is easier to perform operations on it 

cv.imshow('img', img)                                   #displaying the original image read and stored in img variable

kernel_1 = np.ones((3,3))                               #creating a matrix of 3 by 3 having all elements as 1
for i in range(len(kernel_1)):                          #looping throughout the rows
    for j in range(len(kernel_1)):                      #looping in the columns
        kernel_1[i][j]=-1                               #changing all elements to -1 
m=len(kernel_1)//2                                      #taking the mid of list and assigning it to m
kernel_1[m][m]=9                                        #changing the central element to 9 to form the final kernel matrix that will be used to sharpen the image

img1 = cv.filter2D(img, -1, kernel_1)                   #using the filter2D function to sharpen the blurred input image to make it a bit clear
# cv.imshow('img_filter2D_sharp', img1)                   #this displays the sharpened image after add weighting the original image


gauss = cv.GaussianBlur(img, (9,9), 0,)                 #defining Gaussianblur along with a 9X9 kernel to perform add weighted function for image sharpening

img2 = cv.addWeighted(img, 1.5, gauss, -0.5, 3)         #this takes the original image and the gaussian blurred image adds weight to sharpen the image
# cv.imshow('img_addweight_sharp', img2)                  #this displays the sharpened image after add weighting the original image 

img = cv.addWeighted(img1, 0.4, img2, 0.6, 0)           #here we are sharpening the image using the addweighted filter taking gaussian blurred and original image as input in 0.6 and 0.4 ratio to combine them to make the final sharpened image
# cv.imshow('img_final_sharp', img)                       #displaying the final sharpened image


img = cv.cvtColor(img, cv.COLOR_BGR2HSV)                #this convertes the sharpened image i.e. from BGR format into HSV format to split the image into 3 layers of Hue, Saturation and Value
h,s,v = cv.split(img)                                   #splitting the image into Hue, Saturation and Value

hist = cv.calcHist([v], [0], None, [256], [0,155])      #this function calculates the intensity of image and number of pixels having that intensity
plt.plot(hist)                                          #this function plots the histogram of intensity of brightness vs the frequency of pixels having that frequency

v = cv.equalizeHist(v)                                  #here we are optimizing the brightness and contrast of the image

hist = cv.calcHist([v], [0], None, [256], [0,256])      #this function calculates the intensity of image and number of pixels having that intensity
plt.plot(hist)                                          #this function plots the histogram of intensity of brightness vs the frequency of pixels having that frequency

img_merge = cv.merge((h, s, v))                         #after optimizing the image brightness and contrast we are merging the layers of the image that we have splitted before
img_eq = cv.cvtColor(img_merge, cv.COLOR_HSV2BGR)       #after merging the layers convert the image from HSV to BGR again

cv.imshow('img_eq', img_eq)                           #after optimizing the image we are showing the optimized image in the img_eq window
# plt.show()                                              #plotting all the histograms
cv.waitKey(0)                                           #after displaying the image wait for this much time(in miliseconds) here 0 represents infinte time  