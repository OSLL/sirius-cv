from derivative_class import Deriv
import cv2 as cv


test_img = cv.imread('test.jpeg', cv.IMREAD_GRAYSCALE)
query_img = cv.imread('query.jpeg', cv.IMREAD_GRAYSCALE)
test = Deriv(test_img, query_img)

kp1, des1, kp2, des2 = test.addkps(test_img, query_img)
res_img = test.comparekps(test_img, query_img, kp1, des1, kp2, des2)
