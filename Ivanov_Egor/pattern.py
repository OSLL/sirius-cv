from typing import Tuple, Any
from abc import ABC, abstractmethod

import numpy
import cv2


class AnalyzerPattern(ABC):

    def __init__(self, image: numpy.ndarray) -> None:
        self.image: numpy.ndarray = image
        self.keypoints: Any[numpy.ndarray] = None
        self.descriptors: Any[numpy.ndarray] = None

    @abstractmethod
    def plot_points(self) -> Tuple[numpy.ndarray]:
        pass

    @abstractmethod
    def compare_points(self, image,
                       method=cv2.NORM_HAMMING, cross_check=True):
        pass
