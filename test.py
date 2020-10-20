from image_analyze import ImageAnalyze
import cv2
from matplotlib import pyplot as plt
import numpy as np

img1 = cv2.imread('stop.png',0)          # queryImage
img2 = cv2.imread('road.jpg',0) # trainImage
images = ImageAnalyze(img1, img2)
points = images.get_points()
matched_image = images.match_images(points)
resized = cv2.resize(matched_image, None, fx=0.5, fy=0.5)
cv2.imshow('image', resized)
cv2.waitKey(0)