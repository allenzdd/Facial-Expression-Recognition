# face model loading
import tensorflow as tf
import argparse
import json
from utils_hhh import get_yolo_boxes, makedirs, preprocess_input, split_batch_boxes, remove_duplication_boxs
from b_box import draw_boxes, write2txt
from tqdm import tqdm
import numpy as np
import cv2
from emotion_analysis_small import emotion_analysis

# need change the name not YOLO
class YOLO_detect_face(object):
    def __init__(self):
        # facial expression init model
        self.emotion_analysis_small = emotion_analysis()
        self.detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        self.face_detection = cv2.CascadeClassifier(self.detection_model_path)

    def face_detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        return faces


    def draw_face_emotion(self, faces, frame):
        if len(faces) > 0:
            (fX, fY, fW, fH) = faces
            global sum_frame_emotion_dict__
            sum_frame_emotion_dict = {}
            # emotion_dict = {}
            global emotion_dict
            global frame_emotion_dict
            # frame_emotion_dict = {}
            # for box in boxs:
            m, n, __ = np.shape(frame)
            face_img = frame[fY:fY + fH, fX:fX + fW]
            emotion_dict, dominant_emotion, frame_emotion_dict = self.emotion_analysis_small.predict(face_img)
            for emotion in frame_emotion_dict:
                sum_frame_emotion_dict[emotion] = frame_emotion_dict.get(emotion, 0) + sum_frame_emotion_dict.get(emotion, 0)
            sum_frame_emotion_dict__ = dict(sum_frame_emotion_dict)
            # draw rectangle and emotion text
            cv2.rectangle(img=frame, pt1=(fX, fY), pt2=(fW+fX, fH+fY), color=[255,0,0], thickness=5)
            cv2.putText(frame, str(dominant_emotion),(fX, fY),0, 5e-3 * 200, (0,255,0),2)
            return emotion_dict, frame_emotion_dict, sum_frame_emotion_dict__
        else:
            return {}, {}, {}