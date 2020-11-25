from rsd_class import RoadSignDetector
import cv2
import os

img_test = cv2.imread('arrow.png')
img_test = cv2.Canny(img_test, 100, 200)
cv2.imshow("train", img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
rsd = RoadSignDetector(img_test, None)

input_frames_names = os.listdir("test_set/")

input_frames = []
for i, file in enumerate(input_frames_names[:10]):
	print("Frame", i, "from", len(input_frames_names))
	img_raw = cv2.imread("test_set/"+file)
	img = cv2.Canny(img_raw, 100, 200)
	rsd.changeQueryImage(img)
	out = rsd.run()
	out = (out[:,:,None].astype(img.dtype))
	input_frames.append(out)

rsd.createVideo(input_frames)