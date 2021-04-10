
import cv2
import os
import argparse
import os.path
import pathlib
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--MP4Dir", type = pathlib.Path)
parser.add_argument("--OutputDir", type = pathlib.Path)
parser.add_argument("--ImgOutInterval", type = float)
args = parser.parse_args()

cap = cv2.VideoCapture(str(args.MP4Dir))

def getFrame(sec):
    try:
        if not os.path.exists(args.OutputDir):
            os.makedirs(args.OutputDir)

    except OSError:
            print('Unable to directory')

    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames, image = cap.read()
    if hasFrames:
        cv2.imwrite(str(args.OutputDir) + "/image" + str(count) + ".jpg", image)

    return hasFrames
sec = 0

frameRate = args.ImgOutInterval
"""images each n second(s), use decimal for multiple images per second
Example: 2 will an create an image once every 2 seconds"""

count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
