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


class YOLO_detect_face(object):
    def __init__(self):
        self.net_h = 416
        self.net_w = 416
        self.obj_thresh = 0.5
        self.nms_thresh = 0.45 
        self.anchors = [55, 69, 75, 234, 133, 240, 136, 129, 142,
                363, 203, 290, 228, 184, 285, 359, 341, 260]
        self.model_path = "output_graph.pb" 
        self.graph = self._load_graph(self.model_path)
        self.x = self.graph.get_tensor_by_name('prefix/input_1:0')
        self.y0 = self.graph.get_tensor_by_name('prefix/k2tfout_0:0')
        self.y1 = self.graph.get_tensor_by_name('prefix/k2tfout_1:0')
        self.y2 = self.graph.get_tensor_by_name('prefix/k2tfout_2:0')
        # face detection initial tf session
        self.sess=tf.Session(graph=self.graph)

        # facial expression init model
        self.emotion_analysis_small = emotion_analysis()

    def _load_graph(self, frozen_graph_filename):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with open(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="prefix",
                op_dict=None,
                producer_op_list=None
            )
        return graph

    def face_detect(self, frame):
        img_h, img_w, _ = frame.shape
        batch_input = preprocess_input(frame, self.net_h, self.net_w)
        inputs = np.zeros((1, self.net_h, self.net_w, 3), dtype='float32')
        inputs[0] = batch_input
        net_output = self.sess.run([self.y0, self.y1, self.y2], feed_dict={self.x: inputs}) 
        batch_boxes = get_yolo_boxes(
            net_output, img_h, img_w, self.net_h, self.net_w, self.anchors, self.obj_thresh, self.nms_thresh)
        boxs = remove_duplication_boxs(batch_boxes[0], ['face'], self.obj_thresh)
        return boxs

    def draw_face_emotion(self, boxs, frame):
        global sum_frame_emotion_dict__
        sum_frame_emotion_dict = {}
        # emotion_dict = {}
        global emotion_dict
        global frame_emotion_dict
        for box in boxs:
            m, n, __ = np.shape(frame)
            if box[1] < 0:
                box[1] = m + box[1]
            if box[0] < 0:
                box[0] = n + box[0]
            face_img = frame[(box[1]):(box[1]+box[3]), (box[0]):(box[0]+box[2])]
            emotion_dict, dominant_emotion, frame_emotion_dict = self.emotion_analysis_small.predict(face_img)
            for emotion in frame_emotion_dict:
                sum_frame_emotion_dict[emotion] = frame_emotion_dict.get(emotion, 0) + sum_frame_emotion_dict.get(emotion, 0)
            sum_frame_emotion_dict__ = dict(sum_frame_emotion_dict)
            # draw rectangle and emotion text
            cv2.rectangle(img=frame, pt1=(box[0], box[1]), pt2=(box[2]+box[0], box[3]+box[1]), color=[255,0,0], thickness=5)
            cv2.putText(frame, str(dominant_emotion),(box[0], box[1]),0, 5e-3 * 200, (0,255,0),2)
        return emotion_dict, frame_emotion_dict, sum_frame_emotion_dict__
