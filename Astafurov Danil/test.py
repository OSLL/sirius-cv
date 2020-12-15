import _sqlite3
from image_analyze import ImageAnalyze
import cv2
from test_video import from_video_to_frames
from matplotlib import pyplot as plt
import numpy as np
import os
from make_video import make_video
import argparse
from sympy import Point, Polygon, Line

SIGNS = {'t': ['signs/robot/t_2.png', 'signs/robot/t_left.png',
               'signs/robot/t_right.png'],
         'left_t': ['signs/robot/new_t.png', 'signs/robot/left_t_left.png',
                    'signs/robot/left_t_rignt.png'],
         'stop': ['signs/robot/stop_test.png', 'signs/robot/stop_test_left.png',
                  'signs/robot/stop_test_right.png']}


# SIGNS = {sign.split('.')[0]: 'signs/robot/' + sign for sign in
#          os.listdir('signs/robot')}

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
            images = ImageAnalyze(cv2.Canny(sign, 300, 300),
                                  cv2.Canny(img, 300, 300))
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


def detect_signs_homography():
    in_video, out_video = parse_video_name()
    imgs = from_video_to_frames(in_video)
    ready_imgs = []
    is_img = True
    # os.mkdir(in_video.split('.')[0])
    i = 0
    while is_img:
        img = next(imgs)
        if img is None:
            is_img = False
            break
        img = cv2.imread('test_1_Trim/221.png')
        is_img = False

        for sign_name in SIGNS:
            print(sign_name)

            sign = cv2.imread(SIGNS[sign_name])
            images = ImageAnalyze(sign, img)
            img = images.update_img(sign_name)
        i += 1
        cv2.imwrite(in_video.split('.')[0] + '/' + str(i) + '.png', img)

    # make_video(ready_imgs, out_video, imgs[0].shape[1], imgs[0].shape[0],
    #            fps=30)


def detect_signs_dbscan():
    in_video, out_video = parse_video_name()
    imgs = from_video_to_frames(in_video)
    is_img = True
    padding = 20
    width, height, fps = next(imgs)
    i = 0
    writer = cv2.VideoWriter(
        filename=out_video,
        fourcc=-1,  # codec
        fps=fps,  # fps
        frameSize=(int(width) * 2, int(height) * 2),
    )

    while is_img:

        img = next(imgs)

        if img is None:
            is_img = False
            break
        img = cv2.resize(img, fx=2.0, fy=2.0, dsize=None)
        print('frame:', i)

        for sign_name in SIGNS:
            old_rectangles = []
            for sign in SIGNS[sign_name]:

                read_sign = cv2.imread(sign)
                images = ImageAnalyze(
                    cv2.resize(read_sign, fx=2.0, fy=2.0, dsize=None), img)

                new_rectangles = images.dbscan_analyze()
                for new_rectangle in new_rectangles:

                    x1, y1, x2, y2 = new_rectangle
                    new_poly = Polygon([x1, y1], [x2, y1], [x2, y2], [x1, y2])

                    for old_rectangle in old_rectangles:

                        x1, y1, x2, y2 = old_rectangle
                        old_poly = Polygon([x1, y1], [x2, y1], [x2, y2],
                                           [x1, y2])

                        if new_poly.intersection(old_poly):
                            break
                    else:
                        x1, y1, x2, y2 = new_rectangle
                        img = cv2.rectangle(img, (int(x1), int(y1)),
                                            (int(x2), int(y2)), (255, 0, 0),
                                            thickness=2)
                        img = cv2.fillPoly(img, [np.array(((int(x1) - padding,
                                                            int(y1)), (
                                                               int(
                                                                   x1) - padding,
                                                               int(y1) - 20), (
                                                               int(
                                                                   x2) + padding,
                                                               int(y1) - 20), (
                                                               int(
                                                                   x2) + padding,
                                                               int(y1))),
                                                          dtype='int32')], 255)
                        img = cv2.putText(img, sign_name,
                                          (int(x1) - padding, int(y1) - 3),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                          (255, 255, 255),
                                          2, cv2.LINE_AA)

                        old_rectangles.append((x1, y1, x2, y2))

        i += 1
        writer.write(img)

    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_signs_dbscan()
