from typing import Tuple, Any
from abc import ABC, abstractmethod

import numpy
import cv2

from config import ROAD_SIGN


class AnalyzerPattern(ABC):

    def __init__(self, image: numpy.ndarray, type_, filename=None) -> None:
        self.type_ = type_
        self.original_image: numpy.ndarray = image.copy()
        self.temporary_image: numpy.ndarray = image.copy()
        self.final_image: numpy.ndarray = image.copy()
        self.keypoints: Any[numpy.ndarray] = None
        self.descriptors: Any[numpy.ndarray] = None

        if type_ == ROAD_SIGN:
            self.filename = filename

    @abstractmethod
    def plot_points(self) -> Tuple[numpy.ndarray]:
        pass

    @abstractmethod
    def compare_points(self, image,
                       method=cv2.NORM_HAMMING, cross_check=True):
        pass
