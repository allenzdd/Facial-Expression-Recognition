import cv2
import numpy as np
import os
from b_box import BoundBox, bbox_iou
from scipy.special import expit
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def _sigmoid(x):
    return expit(x)

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        #print 'cannot convert: ',(boxes[i].ymin - y_offset) / y_scale * image_h
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]
            
            if(objectness <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[row,col,b,5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes

def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    try:
        if (float(net_w)/new_w) < (float(net_h)/new_h):
            new_h = (new_h * net_w)//new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h)//new_h
            new_h = net_h
    except:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

def normalize(image):
    return image/255.
       
def get_yolo_boxes(net_output, img_h, img_w, net_h, net_w, anchors, obj_thresh, nms_thresh):
    #image_h, image_w, _ = images[0].shape
    nb_images           = 1
    # nb_images           = len(images)
    #batch_input         = np.zeros((nb_images, net_h, net_w, 3))

    # preprocess the input
    #for i in range(nb_images):
     #   batch_input[i] = preprocess_input(images[i], net_h, net_w)        

    # run the prediction
    #batch_output = model.predict_on_batch(batch_input)
    batch_boxes  = [None]*nb_images

    for i in range(nb_images):
        yolos = [net_output[0][i], net_output[1][i], net_output[2][i]]
        boxes = []

        # decode the output of the network
        for j in range(len(yolos)):
            yolo_anchors = anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)

        # correct the sizes of the bounding boxes
        # img_h is position high top-down direction
        correct_yolo_boxes(boxes, img_h, img_w, net_h, net_w)
        # suppress non-maximal boxes
        do_nms(boxes, nms_thresh)  
        batch_boxes[i] = boxes
    return batch_boxes   

def split_batch_boxes(batch_boxes):
    orignial_boxes = []
    for i in range(len(batch_boxes[0])):
        # batch_boxes is two demsions
        temp = batch_boxes[0][i]
        # [x, y, w, h]
        # x=xmin, y=ymin, w=(xmax-xmin), h=(ymax-ymin)
        temp_box = [temp.xmin, temp.ymin, (temp.xmax-temp.xmin), (temp.ymax-temp.ymin)]
        orignial_boxes.append(temp_box)
    return orignial_boxes

def remove_duplication_boxs(boxes, labels, obj_thresh,  quiet=True):
    orignial_boxes = []
    for box in boxes:

        # elinminate face smaller than 30X30
        if (box.xmax-box.xmin) < 50 and (box.ymax-box.ymin) < 50:
            continue
        
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
            temp_box = [box.xmin, box.ymin, (box.xmax-box.xmin), (box.ymax-box.ymin)]
            orignial_boxes.append(temp_box)
    return orignial_boxes


def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap   

def plot_histogram_opencv(emotion_dict):
    canvas = np.ones((500, 600, 3), dtype="uint8") * 255
    sum_values = sum(emotion_dict.values())
    # for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
    for i, emotion in enumerate(emotion_dict):
        sum_prob = emotion_dict[emotion]
        # construct the label text
        text = "{}: {:.2f}".format(emotion, sum_prob)

        # draw the label + probability bar on the canvas
       # emoji_face = feelings_faces[np.argmax(preds)]
        prob = sum_prob / sum_values
        w = int(prob * 600)
        cv2.rectangle(canvas, (7, (i * 70) + 5),
                      (w, (i * 70) + 70), (211, 255, 13), -1)
        # cv2 RGB = B, G, R
        cv2.putText(canvas, text, (10, (i * 70) + 43),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (232, 12, 49), 2)
    w_border=cv2.copyMakeBorder(canvas, 5, 5, 5, 5, borderType= cv2.BORDER_CONSTANT, value=[255,255,255])
    border=cv2.copyMakeBorder(w_border, 2, 2, 2, 2, borderType= cv2.BORDER_CONSTANT, value=[0,0,0])
    return border

def face_location_encoding(face_locations, frame):
    face_images = []
    for top, right, bottom, left in face_locations:
        fX = left
        fY = top
        fW = right - left
        fH = bottom - top
        face_img = frame[fY:fY + fH, fX:fX + fW]
        face_images.append(face_img)
    return face_images

class FaceImg:
    def __init__(self):
        self.face_id_total = []

    def save_face_img(self, face_locations, face_id, frame):
        if face_locations:
            if face_id not in self.face_id_total:
                self.face_id_total.append(face_id)
                for top, right, bottom, left in face_locations:
                    # top *= 4
                    # right *= 4
                    # bottom *= 4
                    # left *= 4
                    # (fX, fY, fW, fH) = faces
                    fX = left
                    fY = top
                    fW = right - left
                    fH = bottom - top
                    face_img = frame[fY:fY + fH, fX:fX + fW]
                    face_img_url = 'static/result/person/p_' + str(face_id) + '.jpg'
                    cv2.imwrite(face_img_url, face_img)

