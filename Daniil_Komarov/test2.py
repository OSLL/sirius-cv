from ImgOps import *
import copy as cp
from sklearn.cluster import DBSCAN

query_img = cv.imread('image_t.jpg')
query_img_gray = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)

train_img = cv.imread('image_q.jpg')
train_img_gray = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
query_kps, query_des = sift.detectAndCompute(query_img_gray, None)
train_kps, train_des = sift.detectAndCompute(train_img_gray, None)

index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(query_des, train_des, k=2)
matchesMask = [[0, 0] for _ in range(len(matches))]

good_matches_query, good_matches_train = [], []
for i, (m, n) in enumerate(matches):
    if m.distance < .5 * n.distance:
        matchesMask[i] = [1, 0]
        good_matches_query.append(m)
#         good_matches_train.append(n)
good_matches_query = np.asarray(good_matches_query)
# good_matches_train = np.asarray(good_matches_train)

draw_params = dict(matchColor=(0, 255, 0),
                   matchesMask=matchesMask,
                   flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
matched_img = cv.drawMatchesKnn(query_img, query_kps, train_img, train_kps, matches, None, **draw_params)

ptQuery_ptTrain = {}

for DMatch_q, DMatch_t in zip(good_matches_query, good_matches_query):
    pt_q = query_kps[DMatch_q.queryIdx].pt
    pt_t = train_kps[DMatch_q.trainIdx].pt
    ptQuery_ptTrain[pt_q] = pt_t


# Clustering
clusterized = DBSCAN(eps=50, min_samples=5).fit_predict(list(ptQuery_ptTrain.keys()))

# Dictionary(cluster_pts_q) = { cluster_name: [(x_q, y_q), ...] }
cluster_pts_q = {}
for gp, pt in zip(clusterized, list(ptQuery_ptTrain.keys())):
    if gp == -1: continue
    else:
        if gp not in cluster_pts_q:
            cluster_pts_q[gp] = [pt]
        else:
            cluster_pts_q[gp].append(pt)

cluster_pts_t = cp.deepcopy(cluster_pts_q)
for cluster in cluster_pts_t:
    for i, pt in enumerate(cluster_pts_t[cluster]):
        cluster_pts_t[cluster][i] = ptQuery_ptTrain[pt]

print(len(cluster_pts_q))
for cluster in cluster_pts_q:
    #print(f'NEW CLUSTER! {cluster}')
    
    src = np.float32(cluster_pts_t[cluster]).reshape(-1, 1, 2)
    dst = np.float32(cluster_pts_q[cluster]).reshape(-1, 1, 2)
    #print(f'SRC_PTS: {src}\n -----\n DST_PTS: {dst}\n\n')
    
    M, _ = cv.findHomography(src, dst, cv.RANSAC, 5.)
    #print(f'M = {M}, type-M = {type(M)}\n\n')
    
    h, w = train_img_gray.shape
    #print(f'h = {h}, w = {w}\n\n')
    
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #print(f'PTS = {pts}, type-PTS = {type(pts)}\n\n\n\n\n')
    
    dst = cv.perspectiveTransform(pts, M)
    
    #print(f'DST FOR TRANSFORM{dst}')
    
    res_img = cv.polylines(query_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

cv.imshow("output", res_img)
cv.waitKey(0)