import cv2
from PIL import Image

from analyzer import Analyzer


def show_image_keypoints(analyzer: Analyzer) -> None:

    Image.fromarray(cv2.drawKeypoints(
        analyzer.image, analyzer.keypoints, None
    )).show()


sign_analyzer = Analyzer(cv2.imread("sign.png"))
test_analyzer = Analyzer(cv2.imread("test_image_1.jpg"))

sign_analyzer.plot_points()
test_analyzer.plot_points()

# Расскомментируете следующие строки, чтобы вывести keypoints для
# каждого изображения
# show_image_keypoints(sign_analyzer)
# show_image_keypoints(test_analyzer)

matches = sign_analyzer.compare_points(test_analyzer, method=cv2.NORM_L2)

result_image = cv2.drawMatches(
    sign_analyzer.image, sign_analyzer.keypoints,
    test_analyzer.image, test_analyzer.keypoints, matches,
    flags=2, outImg=None
)

Image.fromarray(result_image, 'RGB').show()
