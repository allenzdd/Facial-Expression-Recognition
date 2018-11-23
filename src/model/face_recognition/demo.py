#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys

sys.path.append(os.getcwd() + '/src/model/face_recognition')

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

from detect_face_fr import YOLO_detect_face
from emotion_analysis_small import emotion_dashboard, calculate_emotion_dict, domain_emotion_with_id_calculating, FaceDashboard
from utils_hhh import FaceImg
import json

warnings.filterwarnings('ignore')

def main(video_url, frame_size=1):
    yolo = YOLO()
    yolo_face = YOLO_detect_face()
   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    
   # deep_sort 
    root_path = os.getcwd()
    model_filename = root_path + '/src/model/face_recognition/model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(video_url)

    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        # w = int(video_capture.get(3)) 
        # h = int(video_capture.get(4))
        w = 640 * frame_size
        h = 480 * frame_size
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('static/result/video/output.avi', fourcc, 30, (w, h))
        list_file = open('static/result/textfile/detection.txt', 'w')
        # json_brief_file = open('static/result/data/brief.json', 'w+')
        frame_index = -1 

    fps = 0.0
    # start the class of calculate emotion with track id
    emotion = calculate_emotion_dict()
    # initial emotion dashboard
    dashboard = emotion_dashboard(size_time=frame_size)
    # initial domain emotion with face id 
    domain_emotion_analysis = domain_emotion_with_id_calculating()
    # domain emotion dict pointer
    domain_dict = domain_emotion_analysis.emotion_dict_with_id
    # init frame id
    frame_id = 0
    # init data frame
    df = pd.DataFrame(columns=['face_id', "frame_id", "domain_emotion", 
            "angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"])
    # init face dashboard
    face_dashboard = FaceDashboard()
    # init face image save
    face_img_save = FaceImg()
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        # frame = cv2.resize(frame, (640*3, 480*3))
        if ret != True:
            break
        t1 = time.time()

        image = Image.fromarray(frame)

        boxs = yolo.detect_image(image) # eg. [[481, 91, 398, 398], [5, 103, 469, 395]]
        features = encoder(frame,boxs) # features eg. the num of boxes matrix
        
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections]) # reshape boxes and float
        scores = np.array([d.confidence for d in detections]) # score confidence to 1 with each box
        # avoid overlapping boxes
        # https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores) # only return non-overlapping index
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        for track in tracker.tracks:
            if track.is_confirmed() and track.time_since_update >1 :
                continue 
            bbox = track.to_tlbr()
            if (int(bbox[1])-50) > 0:
                img_p = frame[(int(bbox[1])-50):(int(bbox[3])), (int(bbox[0])):(int(bbox[2]))]
            else:
                img_p = frame[0:(int(bbox[3])), (int(bbox[0])):(int(bbox[2]))]
            # if shape not have 0, start face detection
            if 0 in img_p.shape:
                continue
            # start face detection model
            face_locations, face_names = yolo_face.face_detect(img_p, track.track_id)

            # compare face_names and track id
            face_id = int(face_names[0]) if face_names and face_names[0] else track.track_id
            # save new face image
            face_img_save.save_face_img(face_locations, face_id, img_p)

            # emotion predict
            emotion_dict, frame_emotion_dict, sum_frame_emotion_dict, domain_emotion = yolo_face.draw_face_emotion(face_locations, face_names, img_p)

            
            # emotion add to dataframe (df)
            if domain_emotion:
                temp_dict = sum_frame_emotion_dict.copy()
                temp_dict['frame_id'] = frame_id
                temp_dict['face_id'] = face_id
                temp_dict['domain_emotion'] = domain_emotion
                df.loc[frame_id] = temp_dict
                # domain emotion accumlation
                domain_emotion_analysis.calculating(domain_emotion, face_id)
                # draw face dashboard
                frame = face_dashboard.face_interal_five(face_locations, img_p, frame, face_id)

            emotion_id = emotion.sum_track_id_emotion_dict(int(face_id), sum_frame_emotion_dict)
            # try:
            #     json_brief_file.read()
            #     json_brief_file.seek(0)
            #     json_brief_file.truncate()
            # except:
            #     pass
            # json_brief_file.write(json.dumps(emotion_id))

            with open('static/result/data/brief.json', 'w+') as f:
                f.write(json.dumps(emotion_id))

            dashboard.draw_canvas(int(face_id), img_p, emotion_id, domain_dict)
            cv2.putText(frame, str(face_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)


            # if face_names and face_names[0]:
            #     emotion_id = emotion.sum_track_id_emotion_dict(int(face_names[0]), sum_frame_emotion_dict)
            #     dashboard.draw_canvas(int(face_names[0]), img_p, emotion_id, domain_dict)
            #     cv2.putText(frame, face_names[0],(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            # else:
            #     emotion_id = emotion.sum_track_id_emotion_dict(track.track_id, sum_frame_emotion_dict)
            #     dashboard.draw_canvas(track.track_id, img_p, emotion_id, domain_dict)
            #     cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            # draw rectangle and put body id text
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            # cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            frame_id += 1

                    
        if writeVideo_flag:
            # save a frame
            frame = cv2.resize(frame, (640*frame_size, 480*frame_size))
            
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')

            # write emotion data by dataframe
            df.to_csv('static/result/data/summary.csv')

        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return
    # video_capture.release()
    # if writeVideo_flag:
    #     out.release()
    #     list_file.close()
    # print('aaaaaaaaaaaaaaaaaaaaaaaa')
    # print('bbbbbbbbbbbbbbbbbbbbbbbbbb')
    # print('ccccccccccccccccccccccccccc')
    # cv2.destroyAllWindows()
    # return 
    

if __name__ == '__main__':
    frame_size = 1
    main("videos/AdamSavage_2008P-480p.mp4", frame_size=frame_size)


    
