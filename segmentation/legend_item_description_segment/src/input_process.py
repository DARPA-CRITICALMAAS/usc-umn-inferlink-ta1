'''
This module filter out the FP bounding boxes 
from the preceding module (legend-item extraction)
The filtering process is 
step 1. group bboxes by columns
step 2. random select several bboxes in each group
        there are two conditions to filter out FP groups
        1. the bbox image has low color var (blank image), var_threshold=100
        2. the bbox image has many texts using ocr, num_char_threshold=8
step 3. If more than half of the selected bboxes meet one condition
        remove the column group
'''

from collections import defaultdict
from utils import choose_symbol_bbox_in_roi, bboxes2points, ocr_bbox
import os
import cv2
import numpy as np
import random
import json
import requests
import time
import re

def generate_pts4group(legend_area, all_symbol_bbox):    
    output_pts = []
    
    x, y, w, h = legend_area
    bbox_list = choose_symbol_bbox_in_roi(all_symbol_bbox, roi_area=legend_area)

    output_pts = bboxes2points(bbox_list)
    return output_pts, bbox_list

def group_by_columns(points, bbox_list, tolerance=40.0):
    columns_pts = defaultdict(list)
    columns_bboxes = defaultdict(list)
    for i, pt in enumerate(points):
        x, y = pt
        # Find a column this point belongs to, considering the tolerance
        found_column = False
        for column_x in columns_pts.keys():
            if abs(column_x - x) <= tolerance:
                columns_pts[column_x].append((x, y))
                columns_bboxes[column_x].append(bbox_list[i])
                new_avg_x = sum([i[0] for i in columns_pts[column_x]]) / float(len(columns_pts[column_x]))
                columns_pts[new_avg_x] = columns_pts.pop(column_x, None)
                columns_bboxes[new_avg_x] = columns_bboxes.pop(column_x, None)
                found_column = True
                break
        if not found_column:
            columns_pts[x].append((x, y))
            columns_bboxes[x].append(bbox_list[i])
    
    return columns_pts, columns_bboxes

def remove_fp_bbox(columns_bbox, map_tif, random_sample=7):
    columns_flag = {}
    for col_cent in columns_bbox.keys():
        random_res = []
        if len(columns_bbox[col_cent]) <= random_sample:
            random_inds = [i for i in range(len(columns_bbox[col_cent]))]
        else:
            random_inds = random.sample(range(0, len(columns_bbox[col_cent])), random_sample) 
        for ind in random_inds:
            rand_bbox = columns_bbox[col_cent][ind]
            x1, y1, x2, y2 = np.array(rand_bbox).astype('int32')
            bbox_img = map_tif[y1:y2, x1:x2]
            img_var = np.var(bbox_img)
#             print(img_var)
            if img_var < 100:
                random_res.append(0)
                continue
                
            ocr_res = ocr_bbox(bbox_img)
            ocr_res_no_spaces = re.sub(r" +", "", ocr_res)
            random_res.append(1 if len(ocr_res_no_spaces) < 1000 else 0)
                                          
        columns_flag[col_cent] = True if sum(random_res) > random_sample//2 else False
    
    refined_columns_bbox = {}
    for col_cent in columns_flag.keys():
        if columns_flag[col_cent]:
            refined_columns_bbox[col_cent] = columns_bbox[col_cent]
    return refined_columns_bbox

