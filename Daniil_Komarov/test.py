import numpy as np
import cv2
import time

img1 = cv2.imread('image_q.png')          # queryImage
img2 = cv2.imread('image_t2.jpg') # trainImage


scale_percent = 200

width = int(img1.shape[1] * scale_percent / 100)
height = int(img1.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)


img1 = cv2.resize(img1, dsize, interpolation=cv2.INTER_NEAREST)

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None)

cv2.imshow("output", img3)
cv2.waitKey(0)