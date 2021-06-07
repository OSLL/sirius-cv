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

    if(os.path.exists(args.output+"/output.json.nc")):
            with open(args.output+"/output.json.nc") as file:
                startNumeration = json.loads(file.read())['startNumeration']
    else: startNumeration = 0

    result = solution.doUserSelect(out, args.output, startNumeration=startNumeration)

    os.remove(args.input)

    if(result[0] == True): #save work
        with open(args.input, "wb") as file:
            pickle.dump(result[1], file, pickle.HIGHEST_PROTOCOL)

        to_write = result[2]

        if(os.path.exists(args.output+"/output.json.nc")):
            with open(args.output+"/output.json.nc") as file:
                nc = json.loads(file.read())
            to_write.update(nc)

        with open(args.output+"/output.json.nc", "w") as file:
            file.write(json.dumps(to_write))

    else:
        to_write = result[1]
        if(os.path.exists(args.output+"/output.json.nc")):
            with open(args.output+"/output.json.nc") as file:
                nc = json.loads(file.read())
            to_write.update(nc)
        with open(args.output+"/output.json", "w") as file:
            file.write(json.dumps(to_write))

else:
    print("Unsupported action! (render/select)")