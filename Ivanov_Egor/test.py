import os
import sys
import argparse

import cv2
from typing import List

from analyzer import ImageAnalyzer, PointsAnalyzer
from config import ROAD_SIGN, IMAGE


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="path to input video")
parser.add_argument("--output", type=str, help="path to output video")
args = parser.parse_args()


signs: List[ImageAnalyzer] = []

print("Start road sign uploading")
index = 0
signs_list = os.listdir(os.path.join(os.curdir, 'test_signs'))
for filename in signs_list:
    image = cv2.imread(f'test_signs\\{filename}')
    image = cv2.resize(image, (image.shape[0] * 2, image.shape[1] * 2))
    signs.append(ImageAnalyzer(image, ROAD_SIGN, filename.split('.')[0]))
    signs[index].plot_points()
    index += 1

    sys.stdout.write(f"\rUploaded {index} of {len(signs)} \r")

sys.stdout.write("\rRoad signs uploading completed\n")

video_capture = cv2.VideoCapture()
video_capture.open(args.input)

w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
video_writer = cv2.VideoWriter(f"{args.output}", fourcc, 25, (w * 2, h * 2))

sys.stdout.write("\rStart video processing\n")
points_analyzer = PointsAnalyzer()
frame_index = 0
sys.stdout.write("\r\r")
while video_capture.isOpened():
    result = video_capture.read()
    if result[0]:
        frame = ImageAnalyzer(cv2.resize(result[1], (w * 2, h * 2)), IMAGE)
        frame.plot_points()
        output_image = frame.final_image.copy()
        for sign in signs:
            roadsigns = points_analyzer.find_roadsigns(frame, sign)
            if roadsigns is not None:
                for road_sign in roadsigns:
                    x_center = int(sum(road_sign[:2]) / 2)
                    y_center = int(sum(road_sign[2:]) / 2)
                    add_size = int(max(
                        (road_sign[1] - road_sign[0]),
                        (road_sign[3] - road_sign[2]))
                    )
                    output_image = cv2.putText(
                        cv2.rectangle(
                            output_image,
                            (x_center - add_size, y_center - add_size),
                            (x_center + add_size, y_center + add_size),
                            (255, 0, 0), 2
                        ),
                        sign.filename,
                        (x_center - add_size, y_center - add_size),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2
                    )
        video_writer.write(output_image)

    if result[1] is None:
        break

    frame_index += 1
    sys.stdout.write(f"\rProcessed {frame_index} frames\r")

video_capture.release()
video_writer.release()
sys.stdout.write("\rVideo processing completed\r")
