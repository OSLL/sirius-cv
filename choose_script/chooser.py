import re
import os
import sys
import pathlib
import argparse
from typing import List, Union

import cv2
import numpy as np

from linear_markup import LinearMarkup


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument(
    'input_path', type=str, nargs='?',
    help='path to input data (path to folder, video)'
)
argument_parser.add_argument(
    '--output_path', type=str, default=r'.\linear_markup_results',
    help='path to output folder (if path does not exist, it will be created)'
)
argument_parser.add_argument(
    '--skip', type=int, default=24,
    help='count of frames, that will be skipped'
)
arguments = argument_parser.parse_args()

input_path = pathlib.Path(os.path.abspath(arguments.input_path))
if arguments.output_path is not None:
    output_path = pathlib.Path(os.path.abspath(arguments.output_path))
    if not output_path.exists():
        output_path.mkdir()

if not input_path.exists():
    raise Exception('input path does not exist')

videos_paths = []
images_paths = []

videos_captures: List[cv2.VideoCapture] = []

files_count = len(tuple(input_path.iterdir()))
if input_path.is_dir():
    for i, path in enumerate(input_path.iterdir()):
        if re.search(r'(\.mov)|(\.mp4)|(\.avi)$', path.name) is not None:
            videos_paths.append(path)
        elif re.search(r'(\.jpeg)|(\.jpg)|(\.png)$', path.name) is not None:
            images_paths.append(path)
# elif re.search(r'(\.mov)|(\.mp4)|(\.avi)$', input_path.name) is not None:
#     videos_captures.append(cv2.VideoCapture(input_path.name))

linear_markup = LinearMarkup()
for video_path in videos_paths:
    if video_path.name == '2.mp4':
        frame_index = 0
        video_capture = cv2.VideoCapture()
        print(video_path.absolute())
        video_capture.open(str(video_path.absolute()))
        length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while video_capture.isOpened():
            result, frame = video_capture.read()
            if result:
                # if (frame_index + 1) % arguments.skip == 0:
                linear_markup.img_markup(frame, frame_index)

            frame_index += 1
            sys.stdout.write(f'\rAnalyzed {frame_index}/{length} frames.\r')
        video_capture.release()


