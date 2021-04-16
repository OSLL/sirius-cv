import re
import os
import pathlib
from argparse import ArgumentParser
from typing import List

import cv2 as cv

# from linear_markup import LinearMarkup


class LinearMarkup: # заглушка для класса, чтобы с sift'ом не возиться
    def img_markup(self, i, j):
        print("WORK")


VIDEO_REGEX = re.compile(r'(\.mov)|(\.mp4)|(\.avi)$')
IMAGE_REGEX = re.compile(r'(\.jpeg)|(\.jpg)|(\.png)$')


arg_parser = ArgumentParser()
arg_parser.add_argument(
    'input_path', type=str, nargs='?',
    help='path to input data (path to folder, video)'
)
arg_parser.add_argument(
    '--output_path', type=str, default='linear_markup_results',
    help='path to output data (if path does not exist, it will be created)'
)
arg_parser.add_argument(
    '--signs_path', type=str, default='sighs', help='path to signs images'
)
arg_parser.add_argument('--skip', type=int, default=24)
args = arg_parser.parse_args()

input_path = pathlib.Path(os.path.abspath(args.input_path))
if not input_path.exists():
    raise Exception('input path does not exist')

if args.output_path is not None:
    output_path = pathlib.Path(os.path.abspath(args.output_path))
    if not output_path.exists():
        output_path.mkdir()

videos_paths: List[pathlib.Path] = []
images_paths: List[pathlib.Path] = []

if re.search(VIDEO_REGEX, input_path.name) and input_path.is_file():
    videos_paths.append(input_path)
elif input_path.is_dir():
    for filepath in input_path.iterdir():
        if re.search(VIDEO_REGEX, filepath.name) is not None:
            videos_paths.append(filepath)
        elif re.search(IMAGE_REGEX, filepath.name) is not None:
            images_paths.append(filepath)
else:
    raise Exception('unexpected input data type')


linear_markup = LinearMarkup()
video_capture = cv.VideoCapture()
for video_path in videos_paths:
    frame_index = 0
    video_capture.open(str(video_path.absolute()))
    length = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

    while video_capture.isOpened():
        result, frame = video_capture.read()
        if result:
            linear_markup.img_markup(frame, frame_index)
        frame_index += 1

    video_capture.release()

image_index = 0
for image_path in images_paths:
    image = cv.imread(str(image_path.absolute()))
    linear_markup.img_markup(image, image_index)
    image_index += 1
