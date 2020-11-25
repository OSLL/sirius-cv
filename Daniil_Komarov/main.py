from rsd_class import RoadSignDetector
import cv2
import os

img_test = cv2.imread('arrow.png')
#ret, img_test = cv2.threshold(img_test, 127, 255, cv2.THRESH_BINARY)
img_test = cv2.Canny(img_test, 100, 200)
cv2.imshow("train", img_test)
cv2.waitKey(0)
rsd = RoadSignDetector(img_test, None)

input_frames_names = os.listdir("test_set/")

input_frames = []
for i, file in enumerate(input_frames_names[:10]):
	print("Frame", i, "from", len(input_frames_names))
	img = cv2.imread("test_set/"+file)
	#ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
	img = cv2.Canny(img, 100, 200)
	rsd.changeQueryImage(img)
	cv2.imshow("train", rsd.run())
	cv2.waitKey(0)
	input_frames.append(img)

rsd.createVideo(input_frames)