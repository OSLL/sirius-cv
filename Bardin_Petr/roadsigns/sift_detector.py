from .image_processor import ImageProcessor

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import cv2


class SIFTDetector:
    def __init__(self,
                 signs,
                 matcher_index_params=dict(algorithm=0, trees=5),
                 matcher_search_params=dict(checks=50),
                 matcher_knn_k=2,
                 matcher_threshold=0.8):
        self.matcher_threshold = matcher_threshold
        self.matcher_knn_k = matcher_knn_k

        self.sift = cv2.SIFT_create()
        self.matcher = cv2.FlannBasedMatcher(
            matcher_index_params, matcher_search_params)
        self.prepare_signs(signs)

    def compute_kp(self, img):
        return self.sift.detectAndCompute(img, None)

    def prepare_signs(self, signs):
        img, self.signs_size, self.signs_cnt = ImageProcessor.vconcat(
            map(ImageProcessor.prepare_image, signs.values()))
        self.sign_i = img
        self.sign_k, self.sign_d = self.compute_kp(img)
        self.sign_l = list(signs.keys())

    def match(self, target, base):
        res = self.matcher.knnMatch(target, base, k=self.matcher_knn_k)
        return [i for (i, j) in res if i.distance < self.matcher_threshold * j.distance]

    def group(self, base_k, match, dbscan_eps=40):
        pts = np.float32([(self.sign_k[m.queryIdx].pt,
                           base_k[m.trainIdx].pt) for m in match])
        if len(pts) == 0:
            return {}

        sign_groups = np.int32(pts[:, 0, 1] // self.signs_size)

        res = {}

        for sign_g in range(self.signs_cnt):
            dst = pts[sign_groups == sign_g][:, 1, :]
            if dst.shape[0] == 0:
                continue

            dbs = DBSCAN(eps=dbscan_eps, min_samples=1).fit(dst)
            valid_mask = np.zeros_like(dbs.labels_, dtype=bool)
            valid_mask[dbs.core_sample_indices_] = True
            labels = dbs.labels_
            res[self.sign_l[sign_g]] = [tuple(np.int32(np.average(
                dst[(labels == i) & valid_mask], axis=0))) for i in set(labels)]

        return res

    def __call__(self, img):
        base_img = ImageProcessor.prepare_image(img)
        base_k, base_d = self.compute_kp(base_img)

        match = self.match(self.sign_d, base_d)
        groups = self.group(base_k, match)

        for label, data in groups.items():
            for i in data:
                cv2.circle(base_img, i, 3, (0, 0, 255), 5)
                cv2.putText(
                    base_img, label, (i[0]+10, i[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)

        return base_img, groups
