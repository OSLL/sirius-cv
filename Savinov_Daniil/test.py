import cv2 as cv

from modules.derivative_class_upd import CustomDetector


# standards = glob.glob(os.path.join('standards_resized', '*.PNG'))
# print(standards)
standards = ['train_images/30-speed-limit.jpg']
query_img = cv.imread(r'query_images/query_4.jpg')
# input_video_path, output_video_path = 'videos/2.mp4', 'new-result-of-2-video.mp4'

detector = CustomDetector(standards)
res_img, detected_signs_types = detector.detect_on_image(query_img)
print('\n\n\n', detected_signs_types, '\n\n\n')
cv.imshow('result', res_img)
cv.waitKey(0)
