import pandas as pd
import time
from itertools import groupby
from operator import itemgetter
import numpy as np

def calculate_video_time(data_url, face_id):
    # read csv file
    df = pd.read_csv(data_url)
    del df[df.columns[0]]
    # select the face id data
    frame_lst = np.array(df[df['face_id'] == face_id]['frame_id'])
    # groupy consecutive frame id list
    frame_mm_lst = []
    for k, g in groupby(enumerate(frame_lst), lambda ix : ix[0] - ix[1]):
        temp = list(map(itemgetter(1), g))
        frame_mm_lst.append([min(temp), max(temp)])
    # remove possible error
    frame_mm_lst_new = [frame_mm_lst[0]]
    for lst in frame_mm_lst[1:]:
        if lst[0] - frame_mm_lst_new[-1][1] <= 30:
            frame_mm_lst_new[-1][1] = lst[1]
        else:
            frame_mm_lst_new.append(lst)
    # change the frame id to time
    time_lst = []
    for lst in frame_mm_lst_new:
        # due to frame rate is 30
        start_time = time.strftime("%H:%M:%S", time.gmtime(lst[0] / 30))
        end_time = time.strftime("%H:%M:%S", time.gmtime(lst[1] / 30))
        time_lst.append([start_time, end_time])
    # print(time_lst)
    return time_lst

if __name__ == "__main__":
    data_url = input()
    face_id = int(input())
    print(calculate_video_time(data_url, face_id))