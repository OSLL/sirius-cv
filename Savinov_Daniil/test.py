import glob
import os
from derivative_class_upd import SelfDetector


standards = glob.glob(os.path.join('standards_resized', '*.PNG'))
input_video_path, output_video_path = r'videos\2.mp4', 'new-result-of-2-video.mp4'

detector = SelfDetector(standards)
detector.detect_on_video(input_video_path, output_video_path)
