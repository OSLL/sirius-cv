import cv2 as cv


class Pattern:
	def __init__(self, test_img, query_img):
		self.test_img = test_img
		self.query_img = query_img

	def addkps(self):
		pass

	def comparekps(self, test_kps, test_des, query_kps, query_des,
								method=cv.NORM_HAMMING, crossCheck=True, flags=2):
		pass
