from derivative_class import Deriv
import cv2 as cv


# Reading images
test_img = cv.imread('test.jpeg', 0)
query_img = cv.imread('query.jpeg', 0)

# Creating class object
test = Deriv(test_img, query_img)
# Adding keypoints and descriptors
test_kps, test_des, query_kps, query_des = test.addkps()
# Show test resized image with keypoints
test_kps_img = cv.drawKeypoints(test_img, test_kps, None,
                                color=(0, 0, 255), flags=0)
test_kps_img = cv.resize(test_kps_img, None, fx=0.35, fy=0.35, interpolation=cv.INTER_CUBIC)
cv.imshow('Test image with keypoints', test_kps_img)
cv.waitKey(0)
# Show query image with keypoints
query_kps_img = cv.drawKeypoints(query_img, query_kps, None,
                                 color=(0, 0, 255), flags=0)
cv.imshow('Query image with keypoints', query_kps_img)
cv.waitKey(0)

# Comparing descriptors
res_img = test.comparekps(test_kps, test_des, query_kps, query_des)
# Showing resulting matched and resized image
res_img = cv.resize(res_img, None, fx=0.35, fy=0.35, interpolation=cv.INTER_CUBIC)
cv.imshow('Matched image', res_img)
cv.waitKey(0)
