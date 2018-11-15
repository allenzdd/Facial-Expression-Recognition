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

import face_recognition

# need change the name not YOLO
class YOLO_detect_face(object):
    def __init__(self):

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        self.face_index = 0

        self.known_face_encodings = []
        self.known_face_names = []

        # judge the face and person id with hashmap
        self.prev_id = 0
        self.prev_first_match_index = 0
        self.match_dict = {}

        
        # facial expression init model
        self.emotion_analysis_small = emotion_analysis()
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        
        

    def face_detect(self, frame, track_id=0):
        # global face_locations
        # Resize frame of video to 1/4 size for faster face recognition processing
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        small_frame = frame

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(
            rgb_small_frame
            # number_of_times_to_upsample=1, 
            # model='cnn'
            )

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = ""

            # If a match was found in known_face_encodings, just use the first one.
            if self.match_dict.get(track_id, -1) != -1:
                name = self.known_face_names[self.match_dict[track_id]]

            elif True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]
            
            else:
                self.face_index += 1
                # # store the frame in the folder
                # url = "./know_people/" + str(self.face_index) + '.jpg'
                # cv2.imwrite(url, frame)

                # new_people = face_recognition.load_image_file(url)
                
                self.known_face_encodings.append(
                    face_recognition.face_encodings(rgb_small_frame)[0]
                )
                # self.known_face_encodings.append(
                #     face_recognition.face_encodings(new_people)[0]
                # )
                name = str(self.face_index)
                self.known_face_names.append(
                    name
                )
                
            if track_id == self.prev_id and not True in matches:
                # dictionary using .get to do
                # dict(track_id: prev_first_match_index)
                self.match_dict[track_id] = min(self.match_dict.get(track_id, float('inf')), self.prev_first_match_index)
                name = self.known_face_names[self.match_dict[track_id]]
            
            print(self.match_dict)
            self.prev_id = track_id

            face_names.append(name)

        return face_locations, face_names


    def draw_face_emotion(self, face_locations, face_names, frame):
        if face_locations:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # top *= 4
                # right *= 4
                # bottom *= 4
                # left *= 4
                # (fX, fY, fW, fH) = faces
                fX = left
                fY = top
                fW = right - left
                fH = bottom - top
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
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # cv2.rectangle(img=frame, pt1=(fX, fY), pt2=(fW+fX, fH+fY), color=[255,0,0], thickness=5)
                cv2.putText(frame, str(dominant_emotion),(fX, fY),0, 5e-3 * 200, (0,255,0),2)
                return emotion_dict, frame_emotion_dict, sum_frame_emotion_dict__, dominant_emotion
        else:
            return {}, {}, {}, ""