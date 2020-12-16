import cv2


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def prepare_image(data: str, do_image_adjustments=True):
        img = cv2.imread(data) if type(data) == str else data
        if do_image_adjustments:
            pass
        return img

    @staticmethod
    def vconcat(data):
        data = list(data)
        size = sum([max(im.shape) for im in data]) // len(data)
        return cv2.vconcat([cv2.resize(x, (size, size), interpolation=cv2.INTER_CUBIC) for x in data]), size, len(data)
