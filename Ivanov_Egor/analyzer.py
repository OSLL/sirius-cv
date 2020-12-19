from typing import Optional

import cv2
import numpy
from sklearn.cluster import DBSCAN

from pattern import AnalyzerPattern
from config import *


class ImageAnalyzer(AnalyzerPattern):

    def __init__(self, image: numpy.ndarray, type_, filename=None):
        super(ImageAnalyzer, self).__init__(image, type_, filename)

    def plot_points(self) -> None:
        gray_image = cv2.cvtColor(self.temporary_image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(edgeThreshold=100)
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        self.keypoints = keypoints
        self.descriptors = descriptors

    def compare_points(
            self, other_image: AnalyzerPattern,
            *args, **kwargs) -> numpy.ndarray:
        flann = cv2.FlannBasedMatcher(INDEX_PARAMETERS, SEARCH_PARAMETERS)
        matches = flann.knnMatch(
            self.descriptors, other_image.descriptors, k=2
        )
        return matches


class PointsAnalyzer:

    def __init__(self):
        pass

    @staticmethod
    def choose_good_points(raw_matches: numpy.ndarray, k) -> Optional[list]:
        good_points = []

        for match in raw_matches:
            if match[0].distance < k * match[1].distance:
                good_points.append(match[0])

        return good_points

    def find_roadsigns(self, image: ImageAnalyzer, sign_image):
        if image.type_ != IMAGE:
            raise Exception(
                "The image is of the wrong type. Desired image type: IMAGE"
            )

        good_points = self.choose_good_points(
            image.compare_points(sign_image), FILTER_PARAMETERS['k']
        )

        if len(good_points) > 5:
            coordinates = numpy.array([
                [image.keypoints[point.queryIdx].pt[0],
                 image.keypoints[point.queryIdx].pt[1]]
                for point in good_points
            ])

            clustering = DBSCAN(
                min_samples=FILTER_PARAMETERS['min_samples'],
                eps=FILTER_PARAMETERS["eps"]).fit(coordinates)
            roadsigns_keypoints = []

            index = 0
            search = True
            while search:
                possible_coords = coordinates[clustering.labels_ == index]. \
                    astype(int)
                if possible_coords.size > 0:
                    roadsigns_keypoints.append(list(possible_coords))
                else:
                    search = False
                index += 1

            road_signs_coordinates = []
            for keypoints in roadsigns_keypoints:
                x_keypoints = [keypoint[0] for keypoint in keypoints]
                y_keypoints = [keypoint[1] for keypoint in keypoints]

                road_signs_coordinates.append(
                    [min(x_keypoints), max(x_keypoints),
                     min(y_keypoints), max(y_keypoints)]
                )
            return road_signs_coordinates
        return None
