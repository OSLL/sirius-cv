import glob
import os

import cv2 as cv
import numpy as np

from derivative_class import HomographyDetector



class SelfDetector(HomographyDetector):
	
	def draw_bounding_boxes(self, cluster_pts_q: dict, query_img: np.ndarray, sign_name: str) -> np.ndarray:
		for pts in cluster_pts_q.values():
			pts.sort(key=lambda x: x[0])
			min_x, max_x = pts[0][0], pts[-1][0]
			pts.sort(key=lambda x: x[1])
			min_y, max_y = pts[0][1], pts[-1][1]
			
			cnt_x, cnt_y = int((min_x + max_x) / 2), int((min_y + max_y) / 2)
			size = int(max(max_x - min_x, max_y - min_y))
			query_img = cv.rectangle(query_img, (cnt_x - size, cnt_y - size), (cnt_x + size, cnt_y + size), 255, 3, cv.LINE_AA)
			query_img = cv.putText(query_img, sign_name, (cnt_x - size, cnt_y - size), cv.FONT_HERSHEY_DUPLEX,
			                     1, (255, 0, 255), 1, cv.LINE_AA)
			
		return query_img
	
	
	def detect_on_image(self, query_img: np.ndarray) -> np.ndarray:
		for sign_name in self.standard_signs:
			train_img = self.standard_signs[sign_name]
			
			query_kps, query_des, train_kps, train_des = self.add_kps(query_img, train_img)
			good_matches = self.match_kps(query_des, train_des, train_img, query_kps, train_kps)
			
			if len(good_matches):
				cluster_pts_q, cluster_pts_t = self.cluster_pts(good_matches, query_kps, train_kps)
				query_img = self.draw_bounding_boxes(cluster_pts_q, query_img, sign_name)
		
		return query_img
