from derivative_class import Detector
from ImgOps import *
import glob
import os


paths = glob.glob(os.path.join('standards', '*.png'))
paths.append(r'train_images\train_4.jpg')

query_img = cv.imread(r'query_images\query_4.jpg')

detector = Detector(paths)
res_img = detector.detect_on_image(query_img)

imshow(res_img)
