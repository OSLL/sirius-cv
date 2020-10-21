from base_image_analyze import BaseImageAnalyze
import cv2


class ImageAnalyze(BaseImageAnalyze):
    def __init__(self, image_1, image_2):
        super().__init__(image_1, image_2)

    def get_points(self):
        orb = cv2.ORB_create()
        detected_object_1 = orb.detectAndCompute(self.image_1, None)
        detected_object_2 = orb.detectAndCompute(self.image_2, None)
        return detected_object_1, detected_object_2

    def match_images(self, detected_objects,
                     method=cv2.NORM_HAMMING, crossCheck=True, depth=10,
                     flags=2):
        detected_object_1, detected_object_2 = detected_objects
        bf = cv2.BFMatcher(method, crossCheck=crossCheck)
        matches = bf.match(detected_object_1[1], detected_object_2[1])
        matches = sorted(matches, key=lambda x: x.distance)
        result = cv2.drawMatches(self.image_1, detected_object_1[0],
                                 self.image_2, detected_object_2[0],
                                 matches[:depth], flags=flags,
                                 outImg=None)
        return result
