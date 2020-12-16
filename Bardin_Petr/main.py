from roadsigns.video_processor import VideoProcessor
from roadsigns.sift_detector import SIFTDetector
from cv2 import VideoCapture

detector = SIFTDetector({"main": "images/main.png", "ped": "images/ped.png", "stop": "images/stop.png"},
                        matcher_threshold=0.55)
video = VideoProcessor(detector)


def callback(ts, frame, data):
    print(ts, data)


video.run(VideoCapture("videos/test.mp4"), callback,
          output_video_file="videos/output.mp4", show_rt_window=True)
