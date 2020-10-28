from derivative_class import Deriv
import cv2 as cv


# Reading images
test_img = cv.imread('test_3.jpg')
query_img = cv.imread('query_3.jpg')

# Creating class object
test = Deriv(test_img, query_img)

# Show resized test image
test_img = cv.resize(test_img, None, fx=0.7, fy=0.7, interpolation=cv.INTER_CUBIC)
cv.imshow('Test image', test_img)
cv.waitKey(0)
cv.destroyAllWindows()

# Show resized query image
query_img = cv.resize(query_img, None, fx=3., fy=3., interpolation=cv.INTER_CUBIC)
cv.imshow('Query image', query_img)
cv.waitKey(0)
cv.destroyAllWindows()

# Multi-matching
res_img = test.draw_multi_match()

# Show resized result image
res_img = cv.resize(res_img, None, fx=0.7, fy=0.7, interpolation=cv.INTER_CUBIC)
cv.imshow('Multi-match', res_img)
cv.waitKey(0)
cv.destroyAllWindows()
