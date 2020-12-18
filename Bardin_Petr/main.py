from roadsigns.video_processor import VideoProcessor
from roadsigns.sift_detector import SIFTDetector
from cv2 import VideoCapture

signs = {
    # "main": "images/main.png",
    # "ped": "images/ped.png",
    "stop": ["images/stop.png", "images/stop2.png"],
    "left": "images/left.png"
}

detector = SIFTDetector(signs,
                        matcher_threshold=0.7,
                        # upscale_signs=100,
                        # upscale_input=(720, 620)
                        dbscan_eps=100,
                        dbscan_samples=3
                        )
video = VideoProcessor(detector)


def callback(ts, frame, data):
    print(ts, data)


video.run(VideoCapture("videos/test2.mp4"), callback,
          output_video_file="videos/output.mp4", show_rt_window=True)
