from ImgOps import *
import numpy as np
import cv2 as cv


MIN_MATCH_COUNT = 10
query_img = cv.imread('query_4.jpg')  # queryImage
train_img = cv.imread('train_4.jpg')  # trainImage
# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
query_kps, query_des = sift.detectAndCompute(query_img, None)
train_kps, train_des = sift.detectAndCompute(train_img, None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict()
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(query_des, train_des, k=2)

# store all the good matches as per Lowe's ratio test.
good_pts = []
for m, n in matches:
    if m.distance < .7 * n.distance:
        good_pts.append(m)

query_pts = np.float32([query_kps[m.queryIdx].pt for m in good_pts]).reshape(-1, 1, 2)
train_pts = np.float32([train_kps[m.trainIdx].pt for m in good_pts]).reshape(-1, 1, 2)
detected, cnt_obj = False, 0
while True:
    try:
        M, mask = cv.findHomography(train_pts, query_pts, cv.RANSAC, 5.)
        matches_mask = mask.ravel().tolist()
        h, w, d = query_img.shape
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        res_img = cv.polylines(query_img, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)
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
