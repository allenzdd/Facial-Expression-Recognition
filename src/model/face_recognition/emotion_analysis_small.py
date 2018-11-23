from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np
from utils_hhh import plot_histogram_opencv, face_location_encoding

import os
root_path = os.getcwd()

class emotion_dict:
    def __init__(self):
        self.emotion_dict = {}

class calculate_emotion_dict:
    def __init__(self):
        self.dict = emotion_dict().emotion_dict

    def sum_track_id_emotion_dict(self, track_id, temp_emotion_dict):
        if track_id not in self.dict:
            self.dict[track_id] = {}
        for emotion in temp_emotion_dict:
            self.dict[track_id][emotion] = \
                self.dict[track_id].get(emotion, 0) + temp_emotion_dict[emotion]
        # print(self.dict)
        return self.dict

class emotion_analysis:
    def __init__(self):
        # parameters for loading data and images
        # self.emotion_model_path = '/home/dzha4889/face_recogonition_application/demo/src/model/face_recognition/pretrained_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        self.emotion_model_path = root_path + "/src/model/face_recognition/pretrained_models/fer2013_mini_XCEPTION.102-0.66.hdf5"
        # hyper-parameters for bounding boxes shape
        # loading models
        self.emotion_classifier = load_model(self.emotion_model_path, compile=False)
        self.EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        self.emotion_dict = dict(zip(self.EMOTIONS, [0]*len(self.EMOTIONS)))

    def predict(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (64, 64))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        preds = self.emotion_classifier.predict(face)[0]
        preds_dict = dict(zip(self.EMOTIONS, preds))
        
        for emotion in preds_dict:
            self.emotion_dict[emotion] = \
                self.emotion_dict.get(emotion, 0) + preds_dict.get(emotion, 0)


        # emotion_probability = np.max(preds)
        max_label = self.EMOTIONS[preds.argmax()]
        print(max_label)
        return self.emotion_dict, max_label, preds_dict
        

class emotion_dashboard:
    def __init__(self, size_time=1):
        size_time *= 3
        self.pre_position_dict = {0:[0, 0]}
        self.blank_img = np.ones((1920*size_time, 640*size_time, 3)) * 255
        self.size = size_time
    
    def draw_canvas(self, track_id, body_img, emotion_id_dict, domain_dict):
        # list node
        p = self.pre_position_dict
        s = self.size
        if track_id not in p:
            p[track_id] = [((10+(track_id-1)*120)-(640-40)*((track_id-1)//5)) * s, \
                                    (10+130*((track_id-1)//5)) * s]

            # body_img_h = body_img.shape[0]
            # body_img_w = body_img.shape[1]
            # resize?????
            resized_img = cv2.resize(body_img, (40 * s, 120 * s))

            # after resize body image, put the image on a white board (替换矩阵值)
            # self.blank_img[p[track_id-1][1]+10: p[track_id][1], \
            #                     p[track_id-1][0]+10: p[track_id][0]] = \

            self.blank_img[p[track_id][1]:p[track_id][1]+120*s, \
                    p[track_id][0]:p[track_id][0]+40*s] = resized_img
        # no if always update by present tracker id
        # total track id sum emotion data
        new_hist_id = plot_histogram_opencv(emotion_id_dict[track_id])
        resized_hist = cv2.resize(new_hist_id, (60*s,60*s))
        
        self.blank_img[p[track_id][1]+60*s:p[track_id][1]+120*s, \
                            p[track_id][0]+50*s:p[track_id][0]+110*s] = \
                                resized_hist
        # draw domain emotion data if domain emotion exist
        try:
            new_hist_cur_id = plot_histogram_opencv(domain_dict[track_id])
            resized_cur_hist = cv2.resize(new_hist_cur_id, (60*s,60*s))
            self.blank_img[p[track_id][1]:p[track_id][1]+60*s, \
                                p[track_id][0]+50*s:p[track_id][0]+110*s] = \
                                    resized_cur_hist
        except:
            pass
        cv2.imwrite('./static/others/canvas.jpg', self.blank_img)

class domain_emotion_with_id_calculating:
    def __init__(self):
        self.init_emotion_lst = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
        self.emotion_dict_with_id = {}
    
    def calculating(self, domain_emotion, face_id):
        # pointer of emotion_dict_with_id
        dic = self.emotion_dict_with_id
        # if id not in dict, create a new sub dict
        if face_id not in dic:
            dic[face_id] = dict.fromkeys(self.init_emotion_lst, 0)

        # add new domain emotion to direct id
        dic[face_id][domain_emotion] += 1

class FaceDashboard:
    def __init__(self, face_size=60, width=640, height=128, size_time=1):
        self.face_size = face_size
        # self.width = width
        # self.height = height
        self.bottom_dashboard = np.ones((height*size_time, width*size_time, 3)) * 255
        self.interal_five_image = []
        self.temp_id_list = []

    def face_interal_five(self, face_locations, body_image, frame, face_id):
        # when the first face image
        if self.interal_five_image == []:
            face_images = face_location_encoding(face_locations, body_image)
            face_image = face_images[0]
            face_image = cv2.resize(face_image, (self.face_size,self.face_size))
            cv2.putText(face_image, str(face_id), (5,5), 0, 5e-3*200, (0,255,0), 1)
            self.interal_five_image = face_image
            self.temp_id_list = [face_id]
        # after 1st face image and id not appearance 
        if face_id not in self.temp_id_list:
            face_images = face_location_encoding(face_locations, body_image)
            face_image = face_images[0]
            face_image = cv2.resize(face_image, (self.face_size, self.face_size))
            cv2.putText(face_image, str(face_id), (5,5), cv2.FONT_HERSHEY_SIMPLEX, 5e-3*200, (0,255,0), 1)
            # justify length less than five
            # if less than 5, just append
            if len(self.temp_id_list) < 5:
                self.temp_id_list.append(face_id)
            # if large than 5, delete list[0], then delte the first face image
            else:
                self.temp_id_list.pop(0)
                self.interal_five_image = self.interal_five_image[:, self.face_size:]
            # total add new face image
            self.interal_five_image = np.concatenate((self.interal_five_image, face_image), axis=1)
        frame[10:(10 + self.face_size), 10:(10 + len(self.temp_id_list)*self.face_size)] = self.interal_five_image
        return frame

        
