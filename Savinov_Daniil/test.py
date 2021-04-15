import os

import cv2 as cv

from modules.derivative_class_upd import CustomDetector


# standards = glob.glob(os.path.join('standards_resized', '*.PNG'))
# print(standards)
# print(os.listdir('standards_resized'))
standards = os.listdir('standards_resized')
# query_video = cv.imread(r'query_images/query_4.jpg')
input_video_path, output_video_path = 'videos/2.mp4', 'new-result-of-2-video.mp4'

def detect_on_video(input_video_path: str, output_video_path: str) -> None:
    cap = cv.VideoCapture(input_video_path)
    w = 2 * int(cap.get(3))
    h = 2 * int(cap.get(4))
    fps = cap.get(5)
    out = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*'mp4v'),fps, (w, h))

    i = 1
    while cap.isOpened():
        ret, query_img = cap.read()
        query_img = cv.resize(query_img, None, fx=2., fy=2.)
        if ret:
            print(f'NEW FRAME: #{i}')
            # out.write(detect_on_image(self.query_img))
            print(query_img)
            res_img = detector.detect_image(query_img)
            cv.imshow('', res_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print(f'END OF FRAME: #{i}\n\n')
            i += 1
        else:
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()

detector = CustomDetector(standards)
detect_on_video(input_video_path, output_video_path)
# res_img = detector.detect_on_image(query_img)
# # print('\n\n\n', detected_signs_types, '\n\n\n')
# cv.imshow('result', res_img)
# cv.waitKey(0)
