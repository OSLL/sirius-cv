from pattern_class import Pattern
import cv2 as cv
import numpy as np
from typing import Tuple


class Deriv(Pattern):
    def __init__(self, test_img: np.ndarray, query_img: np.ndarray) -> None:
        super().__init__(test_img, query_img)

    def add_kps(self) -> Tuple[list, np.ndarray, list, np.ndarray]:
        self.test_img = cv.cvtColor(self.test_img, cv.COLOR_BGR2GRAY)
        self.query_img = cv.cvtColor(self.query_img, cv.COLOR_BGR2GRAY)
        orb = cv.ORB_create()
        test_kps, test_des = orb.detectAndCompute(self.test_img, None)
        query_kps, query_des = orb.detectAndCompute(self.query_img, None)
        return test_kps, test_des, query_kps, query_des

    def draw_single_match(self, test_kps: list, test_des: np.ndarray,
                          query_kps: list, query_des: np.ndarray,
                          method=cv.NORM_HAMMING, crossCheck=True, flags=2):
        bf = cv.BFMatcher(method, crossCheck=crossCheck)
        matches = bf.match(test_des, query_des)
        matches = sorted(matches, key=lambda x: x.distance)
        res_img = cv.drawMatches(self.test_img, test_kps, self.query_img, query_kps,
                                 matches[:10], flags=flags, outImg=None)
        return res_img

    def draw_multi_match(self, threshold: float = 0.8) -> np.ndarray:
        test_img_gray = cv.cvtColor(self.test_img, cv.COLOR_BGR2GRAY)
        self.query_img = cv.cvtColor(self.query_img, cv.COLOR_BGR2GRAY)
        w, h = self.query_img.shape[::-1]
        res = cv.matchTemplate(test_img_gray, self.query_img, cv.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(self.test_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
        return self.test_img
