import cv2


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def prepare_image(data: str, do_image_adjustments=True, target_size=None):
        img = cv2.imread(data) if type(data) == str else data
        if do_image_adjustments:
            pass
        if target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def vconcat(data, target_size=None):
        data = list(data)
        size = target_size if target_size else sum(
            [max(im.shape) for im in data]) // len(data)
        return cv2.vconcat([cv2.resize(x, (size, size), interpolation=cv2.INTER_CUBIC) for x in data]), size, len(data)
