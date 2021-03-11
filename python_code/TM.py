import numpy as np
from PIL import Image
from PIL import ImageFilter
import cv2


def getDarkChannel(img,blockSize,RGB_Atoms):
    img = np.float64(img)
    Min_img = getMinChannel(img,RGB_Atoms)
    image = Image.fromarray(Min_img)
    image = image.filter(ImageFilter.MinFilter(blockSize))
    DarkChannel = np.asarray(image,dtype=np.float64)
    return DarkChannel




def getMinChannel_new(img):
	imgGray = np.zeros((img.shape[0],img.shape[1]),np.float32)
	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
			localMin = 255
			for k in range(0,2):
				if img.item((i,j,k)) < localMin:
					localMin = img.item((i,j,k))
			imgGray[i,j] = localMin
	return imgGray



def getMaxChannel_new(img):
	imgMax = np.zeros((img.shape[0],img.shape[1]),np.float32)
	for i in range(0,img.shape[0]):
		for j in range(0,img.shape[1]):
			localMax = 0
			for k in range(0,2):
				if img.item((i,j,k)) > localMax:
					localMax = img.item((i,j,k))
			imgMax[i,j] = localMax
	return imgMax