#!/usr/bin/env python
# author@pranjal
# # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # example to show how to use kltracker
# will use a image pair(translated by few pixels) and
# calculate the affine transformation between images 
# to superimpose the second image on the first image
# # # # # # # # # # # # # # # # # # # # # # # # # # # #

import cv2
import numpy as np
from klt import lucasKannadeTracker

# load the two images in grayscale
img1 = cv2.imread('Images/image1.jpg',0)
img2 = cv2.imread('Images/image2.jpg',0)

rows , cols = img2.shape

# choose a patch on the first image
t = img1[50:150,50:150]

# calculate initial warp to
# facilitate klt to search in that region
initialWarp = np.float32([[1.0,0.0,-50.0],[0.0,1.0,-50.0]])

# find affine warping
finalWarp = lucasKannadeTracker( t, img2, initialWarp, 0.010, transform = 'AFFINE')

# subtract initial warp
finalWarp[0,2] -= initialWarp[0,2]
finalWarp[1,2] -= initialWarp[1,2]

print finalWarp

# warp entire img2 to superimpose on img1
img = cv2.warpAffine( img2, finalWarp,( cols, rows))
cv2.imwrite('Images/image.jpg', img)

