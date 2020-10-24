import cv2
import numpy

from pattern import AnalyzerPattern


class Analyzer(AnalyzerPattern):

    def __init__(self, image: numpy.ndarray):
        super(Analyzer, self).__init__(image)

    def plot_points(self) -> None:
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray_image, None)
        self.keypoints = keypoints
        self.descriptors = descriptors

    def compare_points(
            self, other_image: AnalyzerPattern, method=cv2.NORM_HAMMING,
            cross_check=True) -> list:
        bf = cv2.BFMatcher_create(method, crossCheck=cross_check)
        return sorted(
            bf.match(self.descriptors, other_image.descriptors),
            key=lambda point: point.distance
        )
