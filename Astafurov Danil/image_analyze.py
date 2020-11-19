from base_image_analyze import BaseImageAnalyze
import cv2
import numpy as np


class ImageAnalyze(BaseImageAnalyze):
    def __init__(self, image_1, image_2):
        super().__init__(image_1, image_2)

    def get_points(self):
        orb = cv2.ORB_create()
        detected_object_1 = orb.detectAndCompute(self.image_1, None)
        detected_object_2 = orb.detectAndCompute(self.image_2, None)
        return detected_object_1, detected_object_2

    def match_images(self, detected_objects,
                     method=cv2.NORM_HAMMING, crossCheck=True, depth=10,
                     flags=2, k_group=0.2, k_error=0.1):
        detected_object_1, detected_object_2 = detected_objects
        bf = cv2.BFMatcher(method, crossCheck=crossCheck)
        matches = bf.match(detected_object_1[1], detected_object_2[1])
        matches = sorted(matches, key=lambda x: x.distance)
        result = cv2.drawMatches(self.image_1, detected_object_1[0],
                                 self.image_2, detected_object_2[0],
                                 matches[:depth], flags=flags,
                                 outImg=None)
        list_kp1 = []
        list_kp2 = []
        rows1 = self.image_1.shape[0]
        cols1 = self.image_1.shape[1]
        rows2 = self.image_2.shape[0]
        cols2 = self.image_2.shape[1]

        for mat in matches[:depth]:
            img2_idx = mat.trainIdx
            (x2, y2) = detected_object_2[0][img2_idx].pt
            list_kp2.append((x2 + cols1, y2))
        points_x = list(map(lambda z: z[0], list_kp2))
        points_y = list(map(lambda z: z[1], list_kp2))
        list_objects = self.group_points(points_x, points_y, k_group)
        list_objects = self.drop_errors(list_objects, k_error=k_error)
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
                          (int(x2 * 1.05), int(y2 * 1.05)), (200, 0, 0), 2)
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
