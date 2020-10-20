from pattern_class import Pattern
import cv2 as cv


class Deriv(Pattern):
    def addkps(self, test_img, query_img):
        test_img = cv.UMat(test_img)
        query_img = cv.UMat(query_img)
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(test_img, None)
        kp2, des2 = orb.detectAndCompute(query_img, None)
        return kp1, des1, kp2, des2

    def comparekps(self, test_img, query_img, kp1, kp2, des1, des2):
        test_img = cv.UMat(test_img)
        query_img = cv.UMat(query_img)
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        res_img = cv.drawMatches(test_img, kp1, query_img, kp2, matches[:10], None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return res_img
