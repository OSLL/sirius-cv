from rsd_class import RoadSignDetector
import cv2
import os
import argparse
import warnings

def pass_func(*args, **kwargs): pass

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return "Err"

if(__name__ == "__main__"):
    warnings.warn = pass_func

    parser = argparse.ArgumentParser(description="OpenCV roadsign detector")
    parser.add_argument("input", type=str, help="Input video file [.mp4]")
    parser.add_argument("output", type=str, help="Output video file")
    parser.add_argument("--sign", type=str, nargs=2, metavar=("name", "path"), action="append", help="Sign to detect (can be used multiple times). [.png]")
    parser.add_argument("-f", type=str, help="Output video framerate (default is 30)")
    parser.add_argument("--draw-kps", type=str2bool, nargs='?', const=True, default=False, help="Draw keypoints on output video")

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

    if(args_ok):
        rsd = RoadSignDetector(args.draw_kps)

        for sign in args.sign:
            img_test = cv2.imread(sign[1])
            img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
            rsd.addTrainImage(img_test, sign[0])

        cap = cv2.VideoCapture(args.input)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        input_frames = []

        current_frame = 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rsd.changeQueryImage(frame)
                out = rsd.run()
                out = (out[:,:,None].astype(frame.dtype))
                input_frames.append(out)
                printProgressBar(current_frame, total_frames)
                current_frame += 1
            except cv2.error:
                break

        print("Writing video...")
        if(framerate == -1): framerate = 30
        rsd.createVideo(input_frames, framerate, output_file_name)