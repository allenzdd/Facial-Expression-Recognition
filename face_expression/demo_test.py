import pandas as pd
import cv2

def main(video):
    video_capture = cv2.VideoCapture(video)
    print(video_capture)
    while True:
        ret, frame = video_capture.read()
        print(frame)
        print(frame.shape)