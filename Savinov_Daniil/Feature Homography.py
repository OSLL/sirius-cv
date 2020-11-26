from ImgOps import *
import numpy as np
import cv2 as cv


MIN_MATCH_COUNT = 10
query_img = cv.imread('query_4.jpg', 0)  # queryImage
imshow(query_img)
train_img = cv.imread('train_4.jpg', 0)  # trainImage
imshow(train_img)
train_img = resize(train_img, .3, .3)
imshow(train_img)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
query_kps, query_des = sift.detectAndCompute(query_img, None)
train_kps, train_des = sift.detectAndCompute(train_img, None)
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(query_des, train_des, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]


# store all the good matches as per Lowe's ratio test.
good_pts = []
for i, (m, n) in enumerate(matches):
    if m.distance < .6 * n.distance:
        matchesMask[i] = [1, 0]
        good_pts.append(m)


draw_params = dict(matchColor=(0, 255, 0),
                   matchesMask=matchesMask,
                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
matched_img = cv.drawMatchesKnn(query_img, query_kps, train_img, train_kps, matches, None, **draw_params)


imshow(matched_img)


query_pts = np.float32([query_kps[m.queryIdx].pt for m in good_pts]).reshape(-1, 1, 2)
train_pts = np.float32([train_kps[m.trainIdx].pt for m in good_pts]).reshape(-1, 1, 2)
detected, cnt_obj = False, 0
while True:
    try:
        M, mask = cv.findHomography(train_pts, query_pts, cv.RANSAC, 5.)
        matches_mask = mask.ravel().tolist()
        h, w = train_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        res_img = cv.polylines(query_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        detected = True
        query_pts_upd = []
        for pt, flag in zip(query_img, matches_mask):
            if flag == 0:
                query_pts_upd.append(pt)
        query_pts = np.float32(query_pts_upd).reshape(-1, 1, 2)
        cnt_obj += 1
    except Exception:
        pass
    else:
        imshow(res_img)
    finally:
        if detected:
            if cnt_obj == 1: print(f'Congratulations! 1 object was found.')
            else: print(f'Congratulations! {cnt_obj} objects were found.')
        else:
            print('No matches found!')
        break
