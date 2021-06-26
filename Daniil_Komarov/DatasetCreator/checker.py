import argparse
import json
import os

parser = argparse.ArgumentParser(description="Dataset checker")
parser.add_argument("input", type=str)

args = parser.parse_args()

with open(args.input+'/'+"output.json") as file:
    markup = json.loads(file.read())

raw_files = os.listdir(args.input)
images = []
waste = []
waste_images = 0
waste_other = 0
classes = {}
total_signs = 0

for file in raw_files:
    if(os.path.splitext(file)[1] != '.png' and file != "output.json"):
        waste.append(args.input+"/"+file)
        waste_other += 1
    elif(file == "output.json"): pass
    else: images.append(file)

for file in images:
    if(file not in markup):
        waste.append(args.input+"/"+file)
        waste_images += 1
    else:
        sign_class = next(iter(markup[file]["regions"]))[:-2]
        total_signs += 1
        if(sign_class in classes): classes[sign_class] += 1
        else: classes[sign_class] = 1

if(waste):
    print("Found", waste_images, "image(s), that not included to dataset markup and", waste_other, "other waste file(s)")
    delete = False
    while(True):
        inp = input("Delete it? [Y/n]")
        if(inp.lower() == 'y' or inp == ""):
            delete = True
            break
        else: break

    if(delete):
        for file in waste:
            os.remove(file)


print("Dataset balance:")
for sign_class in classes.keys():
    print(sign_class, "-", classes[sign_class], "("+str(round(classes[sign_class]/(total_signs/100), 2))+"%)")