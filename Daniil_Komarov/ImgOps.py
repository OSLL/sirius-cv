import cv2 as cv
import numpy as np


def imshow(img: np.ndarray, label='') -> None:
    cv.imshow(label, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize(img: np.ndarray, fx=.5, fy=.5) -> np.ndarray:
    return cv.resize(img, None, fx=fx, fy=fy)


if __name__ == '__main__':
    img1 = cv.imread('road_2.jpg')
    img2 = resize(img1, .8, .8)
    imshow(img2)
