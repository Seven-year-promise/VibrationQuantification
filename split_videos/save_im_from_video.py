import argparse
from matplotlib import pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import MeanShift, estimate_bandwidth
from skimage.feature import hog
from skimage.morphology import skeletonize

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--video_path', type=str, default = './videos_robustness/',
                   help='sum the integers (default: find the max)')
parser.add_argument('--save_path', type=str, default = './splited_images/',
                   help='sum the integers (default: find the max)')
args = parser.parse_args()

if __name__ == '__main__':
    base_video_path = args.video_path
    video_files = os.listdir(base_video_path)
    im_save_path = args.save_path +  '/'

    video_cnt = 0
    for vfile in video_files:
        if vfile[-3:] != 'avi':
            continue

        video_cnt += 1
        v_path = base_video_path + vfile
        print(v_path)
        cap = cv2.VideoCapture(v_path)
        fame_id = 0
        success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"
        last_frame = frame
        while success:
            last_frame = frame
            success, frame = cap.read()  # "/home/ws/er3973/Desktop/research_code/TailTouching.avi"
            fame_id += 1

        print()
        cv2.imwrite( im_save_path + vfile[:-4] + '.jpg', last_frame)