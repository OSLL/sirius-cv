from ImgOps import *
from sklearn.cluster import OPTICS


show = False

### IMREADING IMAGES ###
query_img = cv.imread('road_2.jpg')
train_img = cv.imread('stop_2.png')

### SHOW ORIGINAL IMAGES
if show: imshow(query_img)
if show: imshow(train_img)


### BGR TO GRAY
query_gray_img = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
train_gray_img = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)


### CREATE KEYPOINTS AND DESCRIPTORS ###
sift = cv.SIFT_create()
query_kps, query_des = sift.detectAndCompute(query_gray_img, None)
train_kps, train_des = sift.detectAndCompute(train_gray_img, None)


### SHOW IMAGES WITH KEYPOINTS ###
query_kps_img = cv.drawKeypoints(query_img, query_kps, query_gray_img,
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
if show: imshow(query_kps_img)
train_kps_img = cv.drawKeypoints(train_img, train_kps, train_gray_img,
                                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
if show: imshow(train_kps_img)


### CREATE MATCHER ###
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

### MATCHING ###
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(query_des, train_des, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]


### RATIO TEST AS PER LOWE'S PAPER ###
good_descs = []
for i, (m, n) in enumerate(matches):
    if m.distance < .6 * n.distance:
        matchesMask[i] = [1, 0]
        good_descs.append(m)
draw_params = dict(matchColor=(0, 255, 0),
                   matchesMask=matchesMask,
                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
matched_img = cv.drawMatchesKnn(query_img, query_kps, train_img, train_kps, matches, None, **draw_params)


### SHOW MATCHED IMAGE ###
if show: imshow(matched_img)


### CLUSTERING ###
x_pts = [query_kps[des.queryIdx].pt[0] for des in good_descs]
y_pts = [query_kps[des.queryIdx].pt[1] for des in good_descs]
good_coords = []
for x, y in zip(x_pts, y_pts):
    good_coords.append([x, y])

cluster_groups = OPTICS(min_samples=25).fit_predict(good_coords)

group_points = {}
for group, point in zip(cluster_groups, good_coords):
    if group not in group_points:
        group_points[group] = [point]
    else:
        group_points[group].append(point)

centers_min, centers_max = [], []
for group in group_points:
    temp = group_points[group]
    pts_x, pts_y = [pt[0] for pt in temp], [pt[1] for pt in temp]
    min_x, max_x, min_y, max_y = int(min(pts_x) - min(pts_x) * 10 // 100), int(max(pts_x) + max(pts_x) * 10 // 100),\
                                 int(min(pts_y) - min(pts_y) * 10 // 100), int(max(pts_y) + max(pts_y) * 10 // 100)
    centers_min.append((min_x, min_y))
    centers_max.append((max_x, max_y))

for start_pt, end_pt in zip(centers_min, centers_max):
    query_img = cv.rectangle(query_img, start_pt, end_pt, (255, 0, 0), 3)

for pt in good_coords:
    query_img = cv.circle(query_img, (int(pt[0]), int(pt[1])), radius=3, color=(0, 255, 0), thickness=-1)

imshow(resize(query_img, .9, .9))
