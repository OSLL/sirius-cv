from rsd_class import RoadSignDetector
import cv2
import argparse
import pickle
import json
import os
import warnings
import time

def current_ms_time():
    return round(time.time() * 1000)

def _pass_func(self, *args, **kwargs): pass

def processFrame(rsd, frame):
    rsd.changeQueryImage(frame)
    out, processed, markup = rsd.run(create_markup=True)
    return out, processed, markup

def saveFrame(raw_image, processed_image, markup, folder, filename):
    os.makedirs(folder, exist_ok=True)
    result = [raw_image, processed_image, markup]
    with open(folder+"/"+filename+".dframe", "wb") as file:
            pickle.dump(result, file, pickle.HIGHEST_PROTOCOL)

def processVideo(rsd, video_folder, video, output_folder, vid_num, vids_total, total_found):
    cap = cv2.VideoCapture(video_folder+"/"+video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    current_frame = 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        ret, frame = cap.read()
        try:
            frame, processed, markup = processFrame(rsd, frame)
            if(markup):
                for proc in processed:
                    saveFrame(frame, proc, markup, output_folder, str(total_found))
                    total_found += 1
            print("Processing video", vid_num, "from", str(vids_total)+";", "Frame", current_frame, "from", str(total_frames)+";", "Found", total_found, "sign(s)")
            #self._printProgressBar(current_frame, total_frames)
            current_frame += 1
        except cv2.error:
            break
    return total_found

def defineSign(rsd, sign):
    img_test = cv2.imread(sign[1])
    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    rsd.addTrainImage(img_test, sign[0])

def generateMarkup(frame_data, sign_name):
    image_filename = args.output+"/"+str(current_ms_time())+".png"
    print(image_filename)
    cv2.imwrite(image_filename, frame_data[0])
    sign_markup = {"fileref": "", "size": os.path.getsize(image_filename), "filename": os.path.basename(image_filename), "base64_img_data": "", "file_attributes": {}, "regions": {}}
    min_x = int(frame_data[2][0][0])
    min_y = int(frame_data[2][0][1])
    max_x = int(frame_data[2][1][0])
    max_y = int(frame_data[2][1][1])
    sign_markup["regions"][sign_name+'-1'] = {"shape-attributes": {"name": "polygon", "all_points_x": [min_x, max_x, min_x, max_x], "all_points_y": [min_y, min_y, max_y, max_y]}, "region-attributes": {}}
    return sign_markup, image_filename

parser = argparse.ArgumentParser(description="OpenCV dataset creator")
parser.add_argument("action", type=str)
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)

args = parser.parse_args()

warnings.warn = _pass_func
rsd = RoadSignDetector(False)

#defineSign(rsd, ("4-WAY-INTERSECT", "signs/4-way-intersect.png"))
#defineSign(rsd, ("DUCK-CROSSING",   "signs/duck-crossing.png"))
defineSign(rsd, ("No-left-turn",    "signs/no-left-turn.png"))
#defineSign(rsd, ("NO-RIGHT-TURN",   "signs/no-right-turn.png"))
#defineSign(rsd, ("PARKING",         "signs/parking.png"))
defineSign(rsd, ("Stop",            "signs/stop.png"))
defineSign(rsd, ("T-intersection",  "signs/t-intersection.png"))
defineSign(rsd, ("traffic-light",   "signs/t-light-ahead.png"))

if(args.action == "render"):
    files = os.listdir(args.input)
    videos = []
    for file in files:
        if(os.path.splitext(file)[1] == ".mp4"):
            videos.append(file)

    vid_num = 1
    vids_total = len(videos)
    total_found = 0

    for video in videos:
        total_found = processVideo(rsd, args.input, video, args.output, vid_num, vids_total, total_found)
        vid_num += 1

elif(args.action == "select"):
    os.makedirs(args.output, exist_ok=True)
    files = os.listdir(args.input)
    frames = []
    for file in files:
        if(os.path.splitext(file)[1] == ".dframe"):
            frames.append(file)

    frames.sort(key=lambda x: int(x[:-7]))
    output_markup = {}
    for frame in frames:
        print("Working on", frame)
        with open(args.input+"/"+frame, "rb") as file:
            frame_data = pickle.load(file)
        cv2.imshow("DatasetCreator", frame_data[1])
        save_and_stop = False
        while(True):
                key_code = cv2.waitKey()
                if(key_code == ord('1')):
                    sign_markup, image_filename = generateMarkup(frame_data, 'Left-T-intersection')
                    print(sign_markup)
                    output_markup[os.path.basename(image_filename)] = sign_markup
                    break
                if(key_code == ord('2')):
                    sign_markup, image_filename = generateMarkup(frame_data, 'T-intersection')
                    print(sign_markup)
                    output_markup[os.path.basename(image_filename)] = sign_markup
                    break
                if(key_code == ord('3')):
                    sign_markup, image_filename = generateMarkup(frame_data, 'Right-T-intersection')
                    print(sign_markup)
                    output_markup[os.path.basename(image_filename)] = sign_markup
                    break

                if(key_code == ord('d')):
                    break
                if(key_code == ord('s')):
                    sign_markup, image_filename = generateMarkup(frame_data, frame_data[2][2])
                    print(sign_markup)
                    output_markup[os.path.basename(image_filename)] = sign_markup
                    break
                if(key_code == ord('e')):
                    save_and_stop = True
                    break
        if(save_and_stop): break
        else: os.remove(args.input+"/"+frame)

    old_data = {}
    if(os.path.exists(args.output+"/output.json")):
        with open(args.output+"/output.json") as file:
            old_data = json.loads(file.read())

    old_data.update(output_markup)
    with open(args.output+"/output.json", "w") as file:
        file.write(json.dumps(old_data))

    print("Output contains", len(old_data), "records")
else:
    print("Unsupported action! (render/select)")