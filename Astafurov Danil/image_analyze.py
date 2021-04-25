import random
from copy import deepcopy
from base_image_analyze import BaseImageAnalyze
import cv2
import numpy as np
from sklearn.cluster import DBSCAN



class ImageAnalyze(BaseImageAnalyze):

    def __init__(self, image_1=None, image_2=None):
        super().__init__(image_1, image_2)

    def get_points(self):
        orb = cv2.ORB_create()
        detected_object_1 = orb.detectAndCompute(self.image_1, None)
        detected_object_2 = orb.detectAndCompute(self.image_2, None)
        return detected_object_1, detected_object_2

    def match_images(self, detected_objects,
                     method=cv2.NORM_HAMMING, crossCheck=True, depth=10,
                     flags=2, k_group=0.2, k_error=0.1, is_match=True):
        detected_object_1, detected_object_2 = detected_objects
        bf = cv2.BFMatcher(method, crossCheck=crossCheck)
        matches = bf.match(detected_object_1[1], detected_object_2[1])
        matches = sorted(matches, key=lambda x: x.distance)
        if is_match:
            result = cv2.drawMatches(self.image_1, detected_object_1[0],
                                     self.image_2, detected_object_2[0],
                                     matches[:depth], flags=flags,
                                     outImg=None)
        else:
            result = self.image_2
        list_kp1 = []
        list_kp2 = []
        rows1 = self.image_1.shape[0]
        cols1 = self.image_1.shape[1]
        rows2 = self.image_2.shape[0]
        cols2 = self.image_2.shape[1]

        for mat in matches[:depth]:
            img2_idx = mat.trainIdx
            (x2, y2) = detected_object_2[0][img2_idx].pt
            list_kp2.append((x2 + (cols1 if is_match else 0), y2))
            cv2.circle(result, (int(x2 + (cols1 if is_match else 0)), int(y2)),
                       5, (255, 0, 0), 3)
        points_x = list(map(lambda z: z[0], list_kp2))
        points_y = list(map(lambda z: z[1], list_kp2))
        list_objects = self.group_points(points_x, points_y, k_group)
        list_objects = self.drop_errors(list_objects, k_error=k_error)
        color = (
            random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        for rect in list_objects:
            rect_x = rect[0]
            rect_y = rect[1]
            x = sum(rect_x) / len(rect_x)
            y = sum(rect_y) / len(rect_y)
            x1 = x - max(max(rect_x) - x, x - min(rect_x))
            y1 = y - max(max(rect_y) - y, y - min(rect_y))
            x2 = x + max(max(rect_x) - x, x - min(rect_x))
            y2 = y + max(max(rect_y) - y, y - min(rect_y))
            # cv2.circle(matched_image, (int(x), int(y)), int(min(max(max(rect_x) - x, x - min(rect_x)), max(max(rect_y) - y, y - min(rect_y))) * 1.2), (200, 0,0), 3)
            # x1, x2 = min(rect_x) - sum(rect_x) / len(rect_x) * 0.05, max(rect_x) + sum(rect_x) / len(rect_x) * 0.05
            # y1, y2 = min(rect_y) - sum(rect_y) / len(rect_y) * 0.05, max(rect_y) + sum(rect_y) / len(rect_y) * 0.05
            cv2.rectangle(result, (int(x1 * 0.95), int(y1 * 0.95)),
                          (int(x2 * 1.05), int(y2 * 1.05)), color, thickness=2)
        return result

    def group_points(self, points_x, points_y, k_group):
        matrix = [[[points_x[0]], [points_y[0]]]]
        for x, y in zip(points_x[1:], points_y[1:]):
            for i in range(len(matrix)):
                mean_point_x = sum(matrix[i][0]) / len(matrix[i][0])
                mean_point_y = sum(matrix[i][1]) / len(matrix[i][1])
                if abs(mean_point_x - x) < mean_point_x * k_group and abs(
                        mean_point_y - y) < mean_point_y * k_group:
                    matrix[i][0].append(x)
                    matrix[i][1].append(y)
                    break
            else:
                matrix.append([[x], [y]])
        return matrix

    def drop_errors(self, rect_matrix, k_error):
        for k, rect in enumerate(rect_matrix):
            restart = True
            rect_x, rect_y = rect
            while restart:
                restart = False
                for i, (x, y) in enumerate(zip(rect_x, rect_y)):
                    if abs(1 - x / np.mean(
                            rect_x[:i] + rect_x[i + 1:])) > k_error:
                        restart = True
                        del rect_x[i]
                        del rect_y[i]
                        break
                    if abs(1 - y / np.mean(
                            rect_y[:i] + rect_y[i + 1:])) > k_error:
                        restart = True
                        del rect_x[i]
                        del rect_y[i]
                        break
            rect_matrix[k] = [rect_x, rect_y]
        return rect_matrix

    def update_img(self, sign_name):
        draw_image = deepcopy(self.image_2)
        result_image = deepcopy(self.image_2)
        while not draw_image is None:
            draw_image, result_image = self.homography_analyze(sign_name, draw_image, result_image)
        return result_image

    def new_analyze(self, img):
        img1 = cv2.imread(img)
        result = deepcopy(img1)
        first = True
        second = True
        for sign in self.signs:
            sign_name = sign
            sign = cv2.imread(self.signs[sign])
            while not img1 is None:
                img2 = deepcopy(img1)
                img1, result = self.update_img(sign_name, sign, img2, result)
            sign = cv2.flip(sign, 1)
            img1 = deepcopy(img2)
            while not img1 is None:
                img2 = deepcopy(img1)
                img1, result = self.update_img(sign_name, sign, img2, result)
        return result

    def homography_analyze(self, sign_name, draw_image, result_image):
        MIN_MATCH_COUNT = 1

        sift = cv2.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.image_1, None)
        kp2, des2 = sift.detectAndCompute(draw_image, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good]).reshape(-1,
                                                            1,
                                                            2)
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good]).reshape(-1,
                                                            1,
                                                            2)
            #print(dst_pts)
            X, Y = sum(dst_pts[:, :, 0]) / len(dst_pts), sum(
                dst_pts[:, :, 1]) / len(dst_pts)
            #print(X, Y)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w, d = self.image_1.shape
            pts = np.float32(
                [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
                -1, 1, 2)
            try:
                dst = cv2.perspectiveTransform(pts, M)
                lines = list(map(lambda e: e[0], np.int32(dst)))
                lines = sorted(sorted(lines, key=lambda e: e[1])[:2],
                               key=lambda e: e[0])
                lines = [[lines[0][0], lines[0][1] - 40],
                         [lines[1][0], lines[0][1] - 40],
                         [lines[1][0], lines[1][1]], [lines[0][0], lines[0][1]]]
                # result_image = cv2.fillPoly(result_image, [np.int32(lines)],
                #                              255)
                # result_image = cv2.putText(result_image, sign_name,
                #                            (lines[-1][0], lines[-1][1] - 20),
                #                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #                            (255, 255, 255),
                #                            2, cv2.LINE_AA)
                result_image = cv2.putText(result_image, sign_name,
                                           (int(X) - 20, int(Y) - 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                           (255, 255, 255),
                                           2, cv2.LINE_AA)
                draw_params = dict(matchColor=(0, 255, 0),
                                   # draw matches in green color
                                   singlePointColor=None,
                                   matchesMask=matchesMask,  # draw only inliers
                                   flags=2)
                result_image = cv2.drawMatches(self.image_1, kp1, result_image, kp2, good, None,
                                      **draw_params)
                # cv2.polylines(result_image, [np.int32(dst)], True, 255,
                #               3, cv2.LINE_AA)
                return (cv2.fillPoly(draw_image, [np.int32(dst)], 255),
                        cv2.circle(result_image, (int(X) + self.image_1.shape[0], int(Y)), 30, (255, 0, 0), thickness=2))
            except:
                return None, result_image
        else:
            return None, result_image

    def dbscan_analyze(self):
        MIN_MATCH_COUNT = 3
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.image_1, None)
        kp2, des2 = sift.detectAndCompute(self.image_2, None)
        rectangles = []
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good.append(m)
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,
                                                                             1,
                                                                             2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,
                                                                             1,
                                                                             2)
            dst_pts = dst_pts.reshape(len(dst_pts), 2)
            model = DBSCAN(eps=50, min_samples=3)
            #print(len(dst_pts))
            y_pred = model.fit_predict(dst_pts)
            #print(y_pred)
            points = [[] for _ in range(max(y_pred) + 1)]
            for i in range(len(dst_pts)):
                if y_pred[i] != -1:
                    points[y_pred[i]].append(dst_pts[i])

            for point_group in points:
                padding = 10
                x1 = min(point_group, key=lambda v: v[0])[0] - padding
                y1 = min(point_group, key=lambda v: v[1])[1] - padding
                x2 = max(point_group, key=lambda v: v[0])[0] + padding
                y2 = max(point_group, key=lambda v: v[1])[1] + padding
                rectangles.append((int(x1), int(y1), int(x2), int(y2)))
            return rectangles

        return rectangles
