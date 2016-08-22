#!/usr/bin/env python
#author@pranjal
# # # # # # # # # # # # # # # # # # # # # # # #
# # # klt tracker # # # # # # # # # # # # # # #
# lucaskannadeTracker function description
# finds the warp which relates img to templt
# ouputWarp(img) -> templt
# outpurWarp applied to img gives templt
# # # # # # # # # # # # # # # # # # # # # # # #
# brief description of function arguments
# wf is a 2x3 matrix for 'AFFINE' transform
# and 3x3 matrix for 'HOMOGRAPHY' transform
# # # # # # # # # # # # # # # # # # # # # # # #

import cv2
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm


def compWarpAffine(w1, w2):
	w = np.empty(shape=(6,1),dtype=w1.dtype)
	w[0,0] = w1[0,0] + w2[0,0] + w1[0,0]*w2[0,0] + w1[2,0]*w2[1,0]
	w[1,0] = w1[1,0] + w2[1,0] + w1[1,0]*w2[0,0] + w1[3,0]*w2[1,0]
	w[2,0] = w1[2,0] + w2[2,0] + w1[0,0]*w2[2,0] + w1[2,0]*w2[3,0]
	w[3,0] = w1[3,0] + w2[3,0] + w1[1,0]*w2[2,0] + w1[3,0]*w2[3,0]
	w[4,0] = w1[4,0] + w2[4,0] + w1[0,0]*w2[4,0] + w1[2,0]*w2[5,0]
	w[5,0] = w1[5,0] + w2[5,0] + w1[1,0]*w2[4,0] + w1[3,0]*w2[5,0]
	return w

def lkAffine(templt, img, wf, epsilon):
	templt = templt.astype('float32')
	rows,cols = templt.shape
	img = img.astype('float32')
	kerX = np.float32([[-0.5,0.0,0.5]])
	kerY = np.float32([-0.5,0.0,0.5])
	gradX = cv2.filter2D(templt,-1,kerX)
	gradY = cv2.filter2D(templt,-1,kerY)

	x = np.empty(shape=(rows,cols),dtype='float32')
	for (r, c), val in np.ndenumerate(x):
		x[r,c] = float(c)
	y = np.empty(shape=(rows,cols),dtype='float32')
	for (r, c), val in np.ndenumerate(y):
		y[r,c] = float(r)

	IxX = np.multiply(gradX,x)
	IxY = np.multiply(gradX,y)
	IyX = np.multiply(gradY,x)
	IyY = np.multiply(gradY,y)
	sgImgs = [IxX,IyX,IxY,IyY,gradX,gradY]

	H = np.empty(shape=(6,6),dtype='float32')
	for (r, c), val in np.ndenumerate(H):
		temp = np.multiply(sgImgs[r],sgImgs[c])
		H[r,c] = np.sum(temp)

	H_INV = inv(H)

	delP = 256*np.ones(shape=(6,1),dtype='float32')

	count = 1

	while(norm(delP,2) > epsilon):

		wImg = cv2.warpAffine(img,wf,(cols,rows))
		errorImg = wImg - templt

		for (r, c), val in np.ndenumerate(delP):
			delP[r,c] = np.sum(np.multiply(sgImgs[r],errorImg))

		delP = np.dot(H_INV,delP)

		dP = (delP)

		dwf = compWarpAffine(np.float32([[wf[0,0]-1.0],[wf[1,0]],[wf[0,1]],[wf[1,1]-1.0],[wf[0,2]],[wf[1,2]]]),dP)

		wf = np.float32([[dwf[0,0]+1.0,dwf[2,0],dwf[4,0]],[dwf[1,0],dwf[3,0]+1.0,dwf[5,0]]])

		#print str(count) + ' : ' + str(norm(delP,2))
		
		count += 1

	return wf



def compWarpHomography(w1, w2):
	w = np.empty(shape=(8,1),dtype=w1.dtype)
	w[0,0] = w1[0,0] + w2[0,0] + w1[0,0]*w2[0,0] + w1[2,0]*w2[1,0] + w1[4,0]*w2[6,0] - w1[6,0]*w2[4,0] - w1[7,0]*w2[5,0]
	w[1,0] = w1[1,0] + w2[1,0] + w1[1,0]*w2[0,0] + w1[3,0]*w2[1,0] + w1[5,0]*w2[6,0]
	w[2,0] = w1[2,0] + w2[2,0] + w1[0,0]*w2[2,0] + w1[2,0]*w2[3,0] + w1[4,0]*w2[7,0]
	w[3,0] = w1[3,0] + w2[3,0] + w1[1,0]*w2[2,0] + w1[3,0]*w2[3,0] + w1[5,0]*w2[7,0] - w1[6,0]*w2[4,0] - w1[7,0]*w2[5,0]
	w[4,0] = w1[4,0] + w2[4,0] + w1[0,0]*w2[4,0] + w1[2,0]*w2[5,0]
	w[5,0] = w1[5,0] + w2[5,0] + w1[1,0]*w2[4,0] + w1[3,0]*w2[5,0]
	w[6,0] = w1[6,0] + w2[6,0] + w1[6,0]*w2[0,0] + w1[7,0]*w2[1,0]
	w[7,0] = w1[7,0] + w2[7,0] + w1[6,0]*w2[2,0] + w1[7,0]*w2[3,0]
	w = w/(1 + w1[6,0]*w2[4,0] + w1[7,0]*w2[5,0])
	return w


def lkHomography(templt, img, wf, epsilon):
	templt = templt.astype('float32')
	rows,cols = templt.shape
	img = img.astype('float32')
	kerX = np.float32([[-0.5,0.0,0.5]])
	kerY = np.float32([-0.5,0.0,0.5])
	gradX = cv2.filter2D(templt,-1,kerX)
	gradY = cv2.filter2D(templt,-1,kerY)

	x = np.empty(shape=(rows,cols),dtype='float32')
	for (r, c), val in np.ndenumerate(x):
		x[r,c] = float(c)
	y = np.empty(shape=(rows,cols),dtype='float32')
	for (r, c), val in np.ndenumerate(y):
		y[r,c] = float(r)

	IxX = np.multiply(gradX,x)
	IxY = np.multiply(gradX,y)
	IyX = np.multiply(gradY,x)
	IyY = np.multiply(gradY,y)
	IxxX = np.multiply(np.multiply(gradX,x),x)
	IxyY = np.multiply(np.multiply(gradY,x),y)
	IxyX = np.multiply(np.multiply(gradX,x),y)
	IyyY = np.multiply(np.multiply(gradY,y),y)
	sgImgs = [IxX,IyX,IxY,IyY,gradX,gradY,-IxyY-IxxX,-IyyY-IxyX]

	H = np.empty(shape=(8,8),dtype='float32')
	for (r, c), val in np.ndenumerate(H):
		temp = np.multiply(sgImgs[r],sgImgs[c])
		H[r,c] = np.sum(temp)

	H_INV = inv(H)

	delP = 256*np.ones(shape=(8,1),dtype='float32')

	count = 1

	while(norm(delP,2) > epsilon):

		wImg = cv2.warpPerspective(img,wf,(cols,rows))
		errorImg = wImg - templt

		for (r, c), val in np.ndenumerate(delP):
			delP[r,c] = np.sum(np.multiply(sgImgs[r],errorImg))

		delP = np.dot(H_INV,delP)

		dP = (delP)

		dwf = compWarpHomography(np.float32([[wf[0,0]-1.0],[wf[1,0]],[wf[0,1]],[wf[1,1]-1.0],[wf[0,2]],[wf[1,2]],[wf[2,0]],[wf[2,1]]]),dP)

		wf = np.float32([[dwf[0,0]+1.0,dwf[2,0],dwf[4,0]],[dwf[1,0],dwf[3,0]+1.0,dwf[5,0]],[dwf[6,0],dwf[7,0],1]])

		#print str(count) + ' : ' + str(norm(delP,2))
		
		count += 1

	return wf


def lucasKannadeTracker( templt, img ,  wf, epsilon, transform = 'AFFINE'):
	if transform == 'AFFINE':
		return lkAffine(templt, img, wf, epsilon)
	elif transform == 'HOMOGRAPHY':
		return lkHomography(templt, img, wf, epsilon)
	else:
		print 'Invalid transform supplied'
