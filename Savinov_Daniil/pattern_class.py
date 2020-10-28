import cv2 as cv
import numpy as np
from abc import abstractmethod
from typing import Tuple


class Pattern:
    def __init__(self, test_img: np.ndarray, query_img: np.ndarray) -> None:
        self.test_img: np.ndarray = test_img
        self.query_img: np.ndarray = query_img

    @abstractmethod
    def add_kps(self) -> Tuple[list, np.ndarray, list, np.ndarray]:
        pass

    @abstractmethod
    def draw_single_match(self, test_kps: list, test_des: np.ndarray,
                          query_kps: list, query_des: np.ndarray,
                          method=cv.NORM_HAMMING, crossCheck=True, flags=2) -> np.ndarray:
        pass

    @abstractmethod
    def draw_multi_match(self, threshold: float = 0.8) -> np.ndarray:
        pass
