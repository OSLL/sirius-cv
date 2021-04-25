import os
import sys
import cv2  # $ pip install opencv-python


def make_video(folder, title, width, height, fps=30, reading=False):

    writer = cv2.VideoWriter(
        filename=title,
        fourcc=-1,  # codec
        fps=fps,  # fps
        frameSize=(int(width), int(height)),
    )
    for img in os.listdir(folder):
        print(img)
        frame = cv2.imread(folder + '/' + img)
        writer.write(frame)
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frames = [f'test_videos/signalAhead_1324866992.avi_image{i}.png' for i in
              range(20)]
    frame = cv2.imread(frames[0])
    make_video(frames, f'output.mov', frame.shape[1], frame.shape[0])
