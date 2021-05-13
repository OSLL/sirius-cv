import re
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import cv2 as cv

from linear_markup_helper import LinearMarkup


VIDEO_REGEX = re.compile(r'(\.avi)|(\.mp4)|(\.mov)|(\.MOV)$')
IMAGE_REGEX = re.compile(r'(\.jpeg)|(\.jpg)|(\.png)|(\.PNG)$')

arg_parser = ArgumentParser('Linear markup script')
arg_parser.add_argument(
    'input_path', type=str, nargs='?',
    help='path to the input data (folder or video)'
)
arg_parser.add_argument(
    '-p', '--signs_path', type=str, default='signs',
    help='path to the signs images'
)
arg_parser.add_argument(
    '-o', '--output_path', type=str, default='linear_markup_results',
    help='path to the output data (if path does not exist, it will be created)'
)
arg_parser.add_argument(
    '-s', '--skip', type=int, default=24,
    help='how many frames will be skipped in the video per each iteration'
)
arg_parser.add_argument(
    '-ss', '--skip_start', type=int, default=0,
    help='how many frames will be skipped in the video per each iteration'
)
args = arg_parser.parse_args()

input_path = Path(args.input_path).absolute()
if not input_path.exists():
    raise Exception('Input path does not exist')

if args.output_path is not None:
    output_path = Path(args.output_path).absolute()
    if not output_path.exists():
        output_path.mkdir()

videos_paths: List[Path] = []
images_paths: List[Path] = []

if re.search(VIDEO_REGEX, input_path.name) and input_path.is_file():
    videos_paths.append(input_path)
elif re.search(IMAGE_REGEX, input_path.name) and input_path.is_file():
    images_paths.append(input_path)
elif input_path.is_dir():
    for filepath in input_path.iterdir():
        if re.search(VIDEO_REGEX, filepath.name) is not None:
            videos_paths.append(filepath)
        elif re.search(IMAGE_REGEX, filepath.name) is not None:
            images_paths.append(filepath)
else:
    raise Exception('Unexpected input data type')

linear_markup = LinearMarkup(args.signs_path)
video_capture = cv.VideoCapture()
for video_path in videos_paths:
    frame_index = 1
    video_capture.open(str(video_path.absolute()))
    length = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

    while video_capture.isOpened():
        result, frame = video_capture.read()
        if result and frame_index % args.skip == 0 and frame_index > args.skip_start:
            scale_percent = 200  # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv.resize(frame, dim, interpolation=cv.INTER_NEAREST)
            linear_markup.img_markup(resized, frame_index)
        frame_index += 1

    video_capture.release()

image_index = 0
for image_path in images_paths:
    image = cv.imread(str(image_path.absolute()))
    linear_markup.img_markup(image, image_index)
    image_index += 1
