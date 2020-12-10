from rsd_class import RoadSignDetector
import cv2
import os

img_test_stop = cv2.imread('a.png')
img_test_stop = cv2.cvtColor(img_test_stop, cv2.COLOR_BGR2GRAY)

img_test_triangle = cv2.imread('b.png')
img_test_triangle = cv2.cvtColor(img_test_triangle, cv2.COLOR_BGR2GRAY)

rsd = RoadSignDetector()
rsd.addTrainImage(img_test_stop, "STOP")
rsd.addTrainImage(img_test_triangle, "TRIANGLE")

cap = cv2.VideoCapture("dt2.mp4")              
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

input_frames = []

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print("Frame", i, "from", len(input_frames_names))
        #img_raw = cv2.imread("test_set/"+file)
        #img = cv2.Canny(img_raw, 100, 200)
        #cv2.imshow("raw", frame)
        rsd.changeQueryImage(frame)
        out = rsd.run()
        #cv2.imshow("output", out)
        out = (out[:,:,None].astype(frame.dtype))
        input_frames.append(out)
        #cv2.waitKey(0)
        print("Working on", i)
        i += 1
    except cv2.error:
        break

rsd.createVideo(input_frames, 15)