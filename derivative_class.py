from pattern_class import Pattern
import cv2 as cv


class Deriv(Pattern):
    def __init__(self, test_img, query_img):
        super().__init__(test_img, query_img)

    def addkps(self):
        orb = cv.ORB_create()
        test_kps, test_des = orb.detectAndCompute(self.test_img, None)
        query_kps, query_des = orb.detectAndCompute(self.query_img, None)
        return test_kps, test_des, query_kps, query_des

    def comparekps(self, test_kps, test_des, query_kps, query_des,
                   method=cv.NORM_HAMMING, crossCheck=True, flags=2):
        bf = cv.BFMatcher(method, crossCheck=crossCheck)
        matches = bf.match(test_des, query_des)
        matches = sorted(matches, key=lambda x: x.distance)
        res_img = cv.drawMatches(self.test_img, test_kps,
                                 self.query_img, query_kps,
                                 matches[:10], flags=flags,
                                 outImg=None)
        return res_img
