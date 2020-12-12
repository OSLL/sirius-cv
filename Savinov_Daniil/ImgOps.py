import cv2 as cv
import numpy as np


def imshow(img: np.ndarray, label='') -> None:
    cv.imshow(label, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize_ratio(img: np.ndarray, fx=.5, fy=.5) -> np.ndarray:
    return cv.resize(img, None, fx=fx, fy=fy)


def resize_tuple(img: np.ndarray, h=500, w=500) -> np.ndarray:
    return cv.resize(img, (h, w))


if __name__ == '__main__':
    img1 = cv.imread('road_2.jpg')
    img2 = resize(img1, .8, .8)
    imshow(img2)
