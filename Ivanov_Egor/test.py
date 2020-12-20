import os
import sys
import argparse

import cv2
from typing import List

from analyzer import ImageAnalyzer, PointsAnalyzer
from config import *


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="path to input video")
parser.add_argument("--output", type=str, help="path to output video")
parser.add_argument("--signs", type=str, help="path to sings")
args = parser.parse_args()


signs: List[ImageAnalyzer] = []

print("Start road sign uploading")
index = 0
signs_list = os.listdir(os.path.join(os.curdir, args.signs))
for filename in signs_list:
    image = cv2.imread(f'{args.signs}\\{filename}')
    image = cv2.resize(
        image, (
            image.shape[0] * SCALE_COEFFICIENT,
            image.shape[1] * SCALE_COEFFICIENT
        )
    )
    signs.append(ImageAnalyzer(image, ROAD_SIGN, filename.split('.')[0]))
    signs[index].plot_points()
    index += 1

    sys.stdout.write(f"\rUploaded {index}/{len(signs)} \r")

sys.stdout.write("\rRoad signs uploading completed\n")

video_capture = cv2.VideoCapture()
video_capture.open(args.input)
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
video_writer = cv2.VideoWriter(
    f"{args.output}", fourcc, fps,
    (w * SCALE_COEFFICIENT, h * SCALE_COEFFICIENT)
)

sys.stdout.write(f"\rProcessing video ({length} frames)\n")
points_analyzer = PointsAnalyzer()
frame_index = 0
while video_capture.isOpened():
    result = video_capture.read()
    if result[0]:
        frame = ImageAnalyzer(
            cv2.resize(
                result[1], (w * SCALE_COEFFICIENT, h * SCALE_COEFFICIENT)
            ), IMAGE
        )
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
    done_percent = round(frame_index / length * 100, 2)
    fill_count = round(done_percent / 4)
    sys.stdout.write(
        f"    |{''.join(['█' * fill_count, '.' * (25 - fill_count)])}|"
        f" - {done_percent}% | ({frame_index}/{length} frames)\r"
    )
    sys.stdout.write("\r\r")

video_capture.release()
video_writer.release()
sys.stdout.write("\nVideo processing completed")
