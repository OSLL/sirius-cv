import _sqlite3
from image_analyze import ImageAnalyze
import cv2
from test_video import from_video_to_frames
from matplotlib import pyplot as plt
import numpy as np
import os
from make_video import make_video
import argparse

SIGNS = {'ped-cross': 'test_photos/1024px-5.png'}
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


def demonstration_video():
    imgs = [('test_photos/40.jpg', 'test_photos/40_test_2.jpg'),
            ('test_photos/ped.jpg', 'test_photos/ped_test.jpg'),
            ('test_photos/road_work.jpg', 'test_photos/road_work_test.jpg')]
    for img, test_img in imgs:
        img1 = cv2.imread(img)  # queryImage
        img2 = cv2.imread(test_img)  # trainImage
        images = ImageAnalyze(img1, img2)
        points = images.get_points()
        print(points)
        detected_object_1, detected_object_2 = points
        # kp1, kp2 = detected_object_1[0], detected_object_2[0]
        matched_image = images.match_images(points, depth=20, k_group=0.3,
                                            k_error=0.3, is_match=False)
        # resized = cv2.resize(matched_image, None, fx=0.3, fy=0.5)
        cv2.imshow('image', matched_image)
        cv2.waitKey(0)

def find_signs():
    signs = os.listdir('signs')
    imgs = from_video_to_frames('output.mov')
    ready_imgs = []
    for img in imgs:
        for sign_name in signs:
            sign = cv2.imread('signs/' + sign_name)  # queryImage
            images = ImageAnalyze(cv2.Canny(sign, 300, 300), cv2.Canny(img, 300, 300))
            points = images.get_points()
            detected_object_1, detected_object_2 = points
            # kp1, kp2 = detected_object_1[0], detected_object_2[0]
            if points[0][0]:
                img = images.match_images(points, depth=20, k_group=0.1,
                                          k_error=0.3, is_match=False)
                # resized = cv2.resize(matched_image, None, fx=0.3, fy=0.5)
                # cv2.imshow('image', img)
                # cv2.waitKey(0)
                print(sign_name)
        ready_imgs.append(img)
    make_video(ready_imgs, 'ready_4.mov', img.shape[1], img.shape[0], fps=1)

def test_2():
    signs = {'ped-cross': 'test_photos/1024px-5.png'}
    images = ImageAnalyze(signs)
    result = images.new_analyze('test_photos/test_ped_4.png')
    cv2.imshow('image', result)
    cv2.waitKey(0)

def parse_video_name():
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--input', type=str, help='Input dir for videos')
    parser.add_argument('--output', type=str, help='Output dir for image')
    args = parser.parse_args()
    return args.input, args.output


def detect_signs():
    in_video, out_video = parse_video_name()
    imgs = from_video_to_frames(in_video)
    ready_imgs = []
    for img in imgs:
        for sign_name in SIGNS:
            sign = cv2.imread(SIGNS[sign_name])  # queryImage
            images = ImageAnalyze(sign, img)
            img = images.update_img(sign_name)
        ready_imgs.append(img)
    make_video(ready_imgs, out_video, img.shape[1], img.shape[0], fps=20)

if __name__ == '__main__':
    detect_signs()