import glob
import json
import os

import cv2 as cv
import numpy as np

from modules.derivative_class_upd import CustomDetector



class LinearMarkup:

    def __init__(self, output_folder_path='linear_markup_results', output_img_format='png'):
        """
        output_folder_path (string, optional):
            Absolute path to the output folder (empty string by default).
        output_img_format (string, optional):
            Format in which images will be saved ('png' format by default).
        """
        self.output_folder_path = output_folder_path
        self.output_img_format = output_img_format
        self.markup_dict = {}

    def img_markup(self, query_img: np.ndarray, i: int) -> None:
        """
        Appending the image to dataset and markup (or not).
        Args:
            query_img (numpy.ndarray): Read image.
            i (integer): Index of current iteration loop.
        """
        # standards - папка с ground-truth изображениями знаков
        standards = glob.glob(os.path.join('standards_resized', '*.png'))
        detector = CustomDetector(standards)
        res_img, markup = detector.detect_image(query_img)
        img_file_name = f'img_{i + 1}.{self.output_img_format}'

        cv.imshow(img_file_name, res_img)
        if cv.waitKey(33) == 50:
            cv.imwrite(
                os.path.join(self.output_folder_path, 'images', img_file_name),
                query_img
            )
            with open(os.path.join(self.output_folder_path, 'markup.json'), 'w') as out_json:
                self.markup_dict.update(
                    {
                        img_file_name: markup
                    }
                )
                json.dump(self.markup_dict, out_json, indent=4)
        cv.destroyAllWindows()
