import numpy as np
import os
import cv2
from colors import get_color
# from src.fermodel import FERModel

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c       = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score      

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3    

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image, boxes, labels, obj_thresh,  quiet=True):
    # facial expression model
    # target_emotions = ['calm', 'anger', 'happiness']
    target_emotions = ['fear', 'surprise', 'anger', 'calm']
    # target_emotions = ['happiness', 'disgust', 'surprise']
    # target_emotions = ['happiness', 'fear', 'disgust', 'calm', 'anger']
    # model_per = FERModel(target_emotions, verbose=True)
    # ==================================================

    nb_box=0
    face_center=[0,0]
    sum_emotion = {}
    for box in boxes:

        # elinminate face smaller than 30X30
       # if (box.xmax-box.xmin) < 50 and (box.ymax-box.ymin) < 50:
       #     continue
        
        # draw the box
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
        # judge the box is a face or not----ZCY
        if label >= 0:
            nb_box+=1
            
            # text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            # width, height = text_size[0][0], text_size[0][1] # this is the text size, a small rectangle
            # region = np.array([[box.xmin-3,        box.ymin], 
            #                    [box.xmin-3,        box.ymin-height-26], 
            #                    [box.xmin+width+13, box.ymin-height-26], 
            #                    [box.xmin+width+13, box.ymin]], dtype='int32')  
            # print(region)
  
            face_center=[(box.xmin+box.xmax)/2,(box.ymin+box.ymax)/2]
            #face=image[max(box.ymin,0):min(box.ymax,image.shape[1]), max(0,box.xmin):min(box.xmax,image.shape[0]), :]
            #img_name=(image_path.split('/')[-1]).split('.')[-2]
            # cv2.rectangle(img=image, pt1=(box.xmin,box.ymin), pt2=(box.xmax,box.ymax), color=get_color(label), thickness=5)
            temp_image = image[(box.ymin-15):(box.ymax+15), (box.xmin-15):(box.xmax+15)]
            # gray_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
            # print('7777777777777777777777')
            # print("%s , %s, %s, %s" %(box.ymin-15, box.ymax+15, box.xmin-15, box.xmax+15))
            # print('7777777777777777777777')
            # calculate emotion dictionary
            # temp_emotion_dict = model_per.predict(temp_image)


            cv2.rectangle(img=image, \
                pt1=(box.xmin - 15, box.ymin - 15), pt2=(box.xmax + 15, box.ymax + 15), \
                                                            color=[255,0,0], thickness=5)
            # print(box.xmin - 15, box.ymin - 15)image
            # print(box.xmax + 15, box.ymax + 15)
            # temp_image = image[(box.xmin-15):(box.xmax+15), (box.ymin-15):(box.ymax+15)]
            # cv2.imwrite('./data/imwrite/' +'_%s' % nb_box + '.jpg', image)
            # cv2.imwrite('./data/imwrite/p' +'_%s' % nb_box + '.jpg', temp_image)
# =============================================================================
#             cv2.fillPoly(img=image, pts=[region], color=get_color(label))
#             cv2.putText(img=image, 
#                         text=label_str, 
#                         org=(box.xmin+13, box.ymin - 13), 
#                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#                         fontScale=1e-3 * image.shape[0], 
#                         color=(0,0,0), 
#                         thickness=2)
# =============================================================================
    # print(temp_emotion_dict)
    return face_center#, temp_emotion_dict   


def write2txt(image, boxes, labels, obj_thresh, imagename, output_path, nb_box, f,quiet=True):
    f.write(imagename)
    f.write(str(nb_box)+'\n')
    for box in boxes:
        
        label_str = ''
        label = -1
        
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                if label_str != '': label_str += ', '
                label_str += (labels[i] + ' ' + str(round(box.get_score()*100, 2)) + '%')
                label = i
            if not quiet: print(label_str)
        # judge the box is a face or not----ZCY
        if label >= 0:
            
            f.write(str(box.xmin)+' '+str(box.ymin)+' '+str(box.xmax-box.xmin)+' '+str(box.ymax-box.ymin)+' '+str(box.get_score())+'\n')

    return 0







