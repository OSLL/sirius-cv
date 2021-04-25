import cv2


class BaseImageAnalyze:
    def __init__(self, image_1, image_2):
        self.image_1 = image_1
        self.image_2 = image_2

    def get_points(self):
        pass

    def match_images(self, detected_objects,
                     method=cv2.NORM_HAMMING, crossCheck=True, depth=10,
                     flags=2):
        pass

