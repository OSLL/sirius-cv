from main import OpenCV_Solution
import cv2
import argparse
import pickle
import json
import os

parser = argparse.ArgumentParser(description="OpenCV dataset creator")
parser.add_argument("action", type=str)
parser.add_argument("input", type=str)
parser.add_argument("output", type=str)

args = parser.parse_args()
solution = OpenCV_Solution()
solution.init()
solution.defineSign(("STOP", "stop.png"))
solution.defineSign(("T-INTERSECT", "t-intersection.png"))

if(args.action == "render"):
    out = solution.linear_markup(args.input)
    with open(args.output+".pkl", "wb") as file:
        pickle.dump(out, file, pickle.HIGHEST_PROTOCOL)

elif(args.action == "select"):
    with open(args.input, "rb") as file:
        out = pickle.load(file)
    os.makedirs(args.output, exist_ok=True)
    with open(args.output+"/output.json", "w") as file:
        file.write(json.dumps(solution.doUserSelect(out, args.output)))
    os.remove(args.input)

else:
    print("Unsupported action! (render/select)")