import glob
import json
import os
import re

import cv2 as cv
import numpy as np

from modules.derivative_class_upd import CustomDetector



def custom_assert(item: object, Error=AssertionError, message='') -> None or Exception:
    if not item:
        raise Error(message)


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
        # self.standards = glob.glob(
        #     os.path.join('standards', '*.png') and
        #     os.path.join('standards', '*.PNG') and
        #     os.path.join('standards', '*.jpg') and
        #     os.path.join('standards', '*.jpeg')
        # )
        self.standards = [
            os.path.abspath(os.path.join(r'..\Savinov_Daniil\standards_resized', file))
            for file in os.listdir(r'..\Savinov_Daniil\standards_resized')
            if
            re.search(r'[a-zA-Z0-9]*((\.png)|(\.PNG)|(\.jpg)|(\.jpeg))', file)
        ]
        print(self.standards)

    def img_markup(self, query_img: np.ndarray, i: int) -> None:
        """
        Append useful images and its markup to dataset, and print the log.
        Args:
            query_img (numpy.ndarray): Read image.
            i (integer): Index of current iteration loop.
        """
        res_img, markup = None, None
        custom_assert(self.standards, FileExistsError, 'No standard images were found')
        detector = CustomDetector(self.standards)
        img_file_name = f'img_{i + 1}.{self.output_img_format}'
        try:
            res_img, markup = detector.detect_image(query_img)
        except Exception:
            custom_assert(res_img, NameError, 'No result image was returned')
            raise ProcessLookupError('Detector cannot detect the image')
        else:
            # if markup['signs']:
            if True:
                cv.imshow(img_file_name, res_img)
                print(markup['signs'])
                key = cv.waitKey(0)
                if key == 50:  # "2" key -- append
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
                    print(f'Image {img_file_name} (#{i + 1}) and markup were successfully writen')  # "2" key
                cv.destroyAllWindows()
                print(f'Image {img_file_name} (#{i + 1}) was skipped')  # "1" key
            print(f'In the image {img_file_name} (#{i + 1}) were no signs detected')  # no signs
        finally:
            print(f'End processing #{i + 1}: {img_file_name}')
