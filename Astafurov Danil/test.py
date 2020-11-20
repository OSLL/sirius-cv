from image_analyze import ImageAnalyze
import cv2
from matplotlib import pyplot as plt
import numpy as np


def f(points_x, points_y):
    matrix = [[[points_x[0]], [points_y[0]]]]
    for x, y in zip(points_x[1:], points_y[1:]):
        for i in range(len(matrix)):
            mean_point_x = sum(matrix[i][0]) / len(matrix[i][0])
            mean_point_y = sum(matrix[i][1]) / len(matrix[i][1])
            if abs(mean_point_x - x) < mean_point_x * 0.3 and abs(
                    mean_point_y - y) < mean_point_y * 0.3:
                matrix[i][0].append(x)
                matrix[i][1].append(y)
                break
        else:
            matrix.append([[x], [y]])
    return matrix


img1 = cv2.imread('stop_2.png', 0)  # queryImage
img2 = cv2.imread('road_2.jpg', 0)  # trainImage
images = ImageAnalyze(img1, img2)
points = images.get_points()
detected_object_1, detected_object_2 = points
# kp1, kp2 = detected_object_1[0], detected_object_2[0]
matched_image = images.match_images(points, depth=30, k_group=0.31, k_error=0.2)
# list_kp1 = []
# list_kp2 = []
# rows1 = img1.shape[0]
# cols1 = img1.shape[1]
# rows2 = img2.shape[0]
# cols2 = img2.shape[1]
#
#
# for mat in matches[:30]:
#     img2_idx = mat.trainIdx
#     (x2, y2) = kp2[img2_idx].pt
#     list_kp2.append((x2 + cols1, y2))
# points_x = list(map(lambda z: z[0], list_kp2))
# points_y = list(map(lambda z: z[1], list_kp2))
# list_objects = f(points_x, points_y)
# print(max(list_objects[1][0]))
# for rect in list_objects:
#     rect_x = rect[0]
#     rect_y = rect[1]
#     x = sum(rect_x) / len(rect_x)
#     y = sum(rect_y) / len(rect_y)
    #cv2.circle(matched_image, (int(x), int(y)), int(min(max(max(rect_x) - x, x - min(rect_x)), max(max(rect_y) - y, y - min(rect_y))) * 1.2), (200, 0,0), 3)
    #x1, x2 = min(rect_x) - sum(rect_x) / len(rect_x) * 0.05, max(rect_x) + sum(rect_x) / len(rect_x) * 0.05
    #y1, y2 = min(rect_y) - sum(rect_y) / len(rect_y) * 0.05, max(rect_y) + sum(rect_y) / len(rect_y) * 0.05
    #cv2.rectangle(matched_image, (int(x -max(max(rect_x) - x, x - min(rect_x))), int(y - max(max(rect_y) - y, y - min(rect_y)))), (int(x + max(max(rect_x) - x, x - min(rect_x))), int(y + max(max(rect_y) - y, y - min(rect_y)))), (200, 0, 0), 2)
resized = cv2.resize(matched_image, None, fx=0.3, fy=0.5)
cv2.imshow('image', resized)
cv2.waitKey(0)
