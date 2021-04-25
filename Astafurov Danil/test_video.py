import time

import cv2

def from_video_to_frames(title):
    frames = []
    cap = cv2.VideoCapture(title)
    yield cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        yield frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    yield None


