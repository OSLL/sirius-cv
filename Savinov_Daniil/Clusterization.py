from ImgOps import *


# Imreading images
query_img = cv.imread('query_4.jpg')
train_img = cv.imread('train_4.jpg')


# Show images
imshow(query_img)
imshow(train_img)


# BGR -> Gray
query_gray_img = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
train_gray_img = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)


# Create descriptors, add keypoints
sift = cv.SIFT_create()
query_kps, query_des = sift.detectAndCompute(query_gray_img, None)
train_kps, train_des = sift.detectAndCompute(train_gray_img, None)


# Show images with keypoints
query_kps_img = cv.drawKeypoints(query_gray_img, query_kps, query_img,
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
imshow(query_kps_img)
train_kps_img = cv.drawKeypoints(train_gray_img, train_kps, train_img,
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
imshow(train_kps_img)


# Create matcher, add matches
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary


# Matching
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(query_des, train_des, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]


# Ratio test as per Lowe's paper: 0.4 - 0.5
good_pts = []
for i, (m, n) in enumerate(matches):
    if m.distance < .7 * n.distance:
        matchesMask[i] = [1, 0]
        good_pts.append(m)
draw_params = dict(matchColor=(0, 255, 0),
                   matchesMask=matchesMask,
                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
matched_img = cv.drawMatchesKnn(query_gray_img, query_kps, train_gray_img, train_kps, matches, None, **draw_params)


# Show matched image
imshow(resize(matched_img, 1.5, 1.5))


# Clustering
# K-Means or OPTICS
