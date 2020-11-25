from pattern_class import RoadSignDetectorPattern
import cv2
import numpy as np
from sklearn.cluster import OPTICS

min_kps = 15

class RoadSignDetector(RoadSignDetectorPattern):
	def __init__(self, test_img, query_img):
			super().__init__(test_img, query_img)

	def detect(self):
		sift = cv2.SIFT_create()
		test_kps, test_des = sift.detectAndCompute(self.test_img, None)
		query_kps, query_des = sift.detectAndCompute(self.query_img, None)
		return test_kps, test_des, query_kps, query_des

	def compare(self, test_kps, test_des, query_kps, query_des):
		flann = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 50})
		matches = flann.knnMatch(test_des, query_des, k=2)
		good_points = []
		for m in matches:
				if m[0].distance < 0.9*m[1].distance:
						good_points.append(m[0])
		self.drawKps(query_kps, good_points)
		print(len(good_points))
		if(len(good_points) > min_kps):
			return good_points
		else:
			return None

	def drawKps(self, query_kps, good_points):
		x_gen = [query_kps[i.trainIdx].pt[0] for i in good_points]
		y_gen = [query_kps[i.trainIdx].pt[1] for i in good_points]
		coords = []
		for i, x_coord in enumerate(x_gen):
			cv2.circle(self.query_img, (int(x_coord), int(y_gen[i])), 15, (255, 0, 0), 2)
		
	def calculateSignCenters(self, query_kps, good_points):
		if(len(good_points) > min_kps):
			x_gen = [query_kps[i.trainIdx].pt[0] for i in good_points]
			y_gen = [query_kps[i.trainIdx].pt[1] for i in good_points]
			coords = []
			for i, x_coord in enumerate(x_gen):
				coords.append([x_coord, y_gen[i]])

			points_to_clusterize = np.array(coords)
			clust = OPTICS(min_samples=min_kps, max_eps=150)
			clust.fit(points_to_clusterize)

			roadsign_kps = []
			i = 0
			while(True):
				points = points_to_clusterize[clust.labels_ == i]
				points = points.astype(int)
				if(points.size != 0):
					temp = []
					for point in points:
						temp.append(point)
					roadsign_kps.append(temp)
				else:
					break
				i += 1
			roadsign_coords = []
			for kps in roadsign_kps:
				x_kps = [i[0] for i in kps]
				y_kps = [i[1] for i in kps]
				roadsign_coords.append([min(x_kps), max(x_kps), min(y_kps), max(y_kps)])
			return roadsign_coords
		else:
			return None
	
	def run(self):
		t_kp, t_des, q_kp, q_des = self.detect()
		success = False
		for i in range(3):
			try:
				good_points = self.compare(t_kp, t_des, q_kp, q_des)
				success = True
				break
			except IndexError:
				pass
		if(success and good_points != None):
			roadsign_centers = self.calculateSignCenters(q_kp, good_points)
			for center in roadsign_centers:
				x_size = center[1]-center[0]
				y_size = center[3]-center[2]
				add_size_x = int(x_size/100*40)
				add_size_y = int(y_size/100*40)
				self.query_img = cv2.rectangle(self.query_img, (center[0]-add_size_x, center[2]-add_size_y), (center[1]+add_size_x, center[3]+add_size_y), (0, 0, 0), 2)
			return self.query_img
		else:
			print("Image doesn't contain road signs!")
			return self.query_img
		
	def changeQueryImage(self, query_img):
		self.query_img = query_img

	def createVideo(self, images):
		height, width, layers = images[0].shape
		video = cv2.VideoWriter("output.mp4", 0, 5, (width, height))
		for frame in images:
			video.write(frame)
		video.release()
