# Road Sign Detector

This program analyzes video and finds user-defined road signs on it.

**Usage:** `py main.py [-h] [--sign NAME PATH] [-f FRAMERATE] [--draw-kps] input output`

## Description of arguments
* position arguments:
  * input - Input video file. Must be a .mp4 file.
  * output - Output video file. Can be specified as .mp4 file or without extension.
* optional arguments:
  * -h, --help - Print help message in console.
  * --sign NAME PATH - Defines sign(s) to detect. NAME - name of that sign, will be specified on resulting video. PATH - path to image file with sign (Must be a .png file). If you want to detect several signs, you need to repeat this key for each sign. 
  * -f FRAMERATE - Defines framerate for output video. If input video framerate and entered here aren't equal video will be converted.
  * --draw-kps - *Debug option* Draw keypoints' circles on resulting video.

## WARNING! THERE MUST BE AT LEAST ONE --sign ARGUMENT
