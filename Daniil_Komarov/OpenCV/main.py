from rsd_class import RoadSignDetector
import cv2
import os
import argparse
import warnings
import json

class OpenCV_Solution():
    def _pass_func(self, *args, **kwargs): pass

    def _printProgressBar(self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()

    def _str2bool(self, v):
        if isinstance(v, bool):
           return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return "Err"

    def _process_args(self):
        parser = argparse.ArgumentParser(description="OpenCV roadsign detector")
        parser.add_argument("input", type=str, help="Input video file [.mp4]")
        parser.add_argument("output", type=str, help="Output video file")
        parser.add_argument("--sign", type=str, nargs=2, metavar=("name", "path"), action="append", help="Sign to detect (can be used multiple times). [.png]")
        parser.add_argument("-f", type=str, help="Output video framerate (default is 30)")
        parser.add_argument("--draw-kps", type=self._str2bool, nargs='?', const=True, default=False, help="Draw keypoints on output video")

        args = parser.parse_args()
        args_ok = True
        framerate = -1

        if(not os.path.exists(args.input)):
            print("Input video file not exists!")
            args_ok = False

        if(os.path.splitext(args.input)[1] != ".mp4"):
            print("Input video file must be .mp4!")
            args_ok = False

        output_file_name = args.output
        if(os.path.splitext(output_file_name)[1] != ".mp4"): output_file_name += ".mp4"
        if(os.path.exists(output_file_name)):
            inp = input("Output file already exists! Do you want to rewrite it? [N/y] ")
            if(inp.lower() != "y"):
                args_ok = False

        if(args.sign != None):
            for sign in args.sign:
                if(not os.path.exists(sign[1])):
                    print("Sign", sign[0], "file not exists!")
                    args_ok=False
                if(os.path.splitext(sign[1])[1] != ".png"):
                    print("Sign", sign[0], "must be .png!")
                    args_ok = False
        else:
            print("There must be at least one --sign")
            args_ok = False

        if(args.f != None):
            if(not args.f.isdigit()):
                print("Framerate must be an integer!")
                args_ok = False
                framerate = int(args.f)

        return args, args_ok, framerate, output_file_name

    def _process_frame(self, frame, create_markup=False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.rsd.changeQueryImage(frame)
        if(create_markup): out, processed, markup = self.rsd.run(create_markup=True)
        else: out, processed = self.rsd.run()
        out = (out[:,:,None].astype(frame.dtype))
        if(create_markup): return out, processed, markup
        else: return out, processed

    def _process_video(self, video_path, create_markup=False, cut_empty=True):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        output_frames = []
        current_frame = 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                if(create_markup):
                    out, processed, markup = self._process_frame(frame, create_markup=True)
                    output_frames.append([out, processed, markup])
                else:
                    out, processed = self._process_frame(frame)
                    output_frames.append([out, processed])
                self._printProgressBar(current_frame, total_frames)
                current_frame += 1
            except cv2.error:
                break
        output = []
        if(cut_empty):
            for frame in output_frames:
                if(frame[2]): output.append(frame)
        else: output = output_frames
        return output

    def detect_image(self, image):
        out, processed, markup = self._process_frame(image, create_markup=True)
        json_markup = {"signs": []}
        for sign in markup:
            left  = int(sign[0][0])
            up    = int(sign[0][1])
            right = int(sign[1][0])
            down  = int(sign[1][1])

            sign_descr = {}
            sign_descr['type'] = sign[2]
            sign_descr['left_down'] = {"x": left, "y": down}
            sign_descr['right_up'] = {"x": right, "y": up}

            json_markup['signs'].append(sign_descr)
        return out, processed, json.dumps(json_markup)

    def linear_markup(self, input_path):
        if(type(input_path) == type(str)): return "ERROR"
        else:
            output_frames = []
            if(os.path.isfile(input_path)):
                output_frames = self._process_video(input_path, create_markup=True)
            else:
                for file in os.listdir(input_path):
                    if(os.path.splitext(file)[1].lower() in ['.jpg', '.jpeg', '.png']):
                        frame = cv2.imread(os.path.join(input_path, file))
                        out, processed, markup = self._process_frame(frame, create_markup=True)
                        output_frames.append([out, processed, markup])
                    if(os.path.splitext(file)[1].lower() in ['.mp4']):
                        for out, processed, markup in self._process_video(os.path.join(input_path, file), create_markup=True):
                            output_frames.append([out, processed, markup])
        return output_frames

    def doUserSelect(self, frames, output_directory, startNumeration=0):
        os.makedirs(output_directory, exist_ok=True)
        output_markup = {}
        for frame in frames:
            frame_markup = {"signs": []}
            cv2.imshow("output", frame[1])
            while(True):
                key_code = cv2.waitKey()
                if(key_code == ord('1')):
                    break
                if(key_code == ord('2')):
                    cv2.imwrite(os.path.join(output_directory, str(startNumeration)+".png"), frame[0])

                    for sign in frame[2]:
                        left  = int(sign[0][0])
                        up    = int(sign[0][1])
                        right = int(sign[1][0])
                        down  = int(sign[1][1])

                        sign_descr = {}
                        sign_descr['type'] = sign[2]
                        sign_descr['left_down'] = {"x": left, "y": down}
                        sign_descr['right_up'] = {"x": right, "y": up}

                        frame_markup['signs'].append(sign_descr)
                    output_markup[str(startNumeration)+".png"] = frame_markup
                    startNumeration += 1
                    break
        return output_markup

    def init(self, draw_kps=False):
        warnings.warn = self._pass_func
        self.rsd = RoadSignDetector(draw_kps)

    def defineSign(self, sign):
        img_test = cv2.imread(sign[1])
        img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
        self.rsd.addTrainImage(img_test, sign[0])

if(__name__ == "__main__"):
    solution = OpenCV_Solution()
    args, args_ok, framerate, output_file_name = solution._process_args()

    if(args_ok):
        solution.init(draw_kps=args.draw_kps)

        for sign in args.sign:
            solution.defineSign(sign)

        output_frames = solution._process_video(args.input)

        print("Writing video...")
        if(framerate == -1): framerate = 30
        solution.rsd.createVideo(output_frames, framerate, output_file_name)