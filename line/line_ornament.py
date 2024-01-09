import os
import cv2
import numpy as np
from helper.process_shp import read_shp
from shapely.geometry import LineString, Point, box
from PIL import Image
import math
import json

def draw_line_in_image(line_list, empty_image, row, col, buffer=50):
    for i in range(1, len(line_list)):
        pt1 = (int(line_list[i-1][1])-col, int(line_list[i-1][0])-row)
        pt2 = (int(line_list[i][1])-col, int(line_list[i][0])-row)
        cv2.line(empty_image, pt1, pt2, (1,1,1), buffer)
    return empty_image

def binarize_image(line_area):
    avg_gray_val = np.average(line_area[np.where(line_area>0)])
    line_area = line_area.astype('uint8')
    line_area_gray = cv2.cvtColor(line_area, cv2.COLOR_BGR2GRAY) # white is bg
    min_gray_val = np.min(line_area_gray)
#     print('--- ', avg_gray_val, min_gray_val)
    binary_thres = (avg_gray_val + min_gray_val) // 2
    _, binary_image = cv2.threshold(line_area_gray, binary_thres, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def project_pt(pt1, pt2, pt):
    # Define the points A, B, and P as NumPy arrays
    A = np.array([pt1[0], pt1[1]])
    B = np.array([pt2[0], pt2[1]])
    P = np.array([pt[0], pt[1]])
    # Step 1: Calculate vector AP
    AP = P - A
    # Step 2: Calculate vector AB
    AB = B - A
    # Step 3: Calculate the projection of AP onto AB
    # This is done by finding the dot product of AP and AB, divided by the dot product of AB with itself
    # Then multiply the unit vector AB by this scalar
    projection_scalar = np.dot(AP, AB) / np.dot(AB, AB)
    projection_vector = projection_scalar * AB
    # Step 4: Find the coordinates of the projection
    projection_point = A + projection_vector
    return (projection_point)

def check_direction(line_shp, symbol_pt):
    line_pt1, line_pt2 = line_shp.coords[0], line_shp.coords[-1]
    projected_pt = project_pt(line_pt1, line_pt2, symbol_pt)
    proj_symbol_pt = (symbol_pt[0] - projected_pt[0]) * (symbol_pt[1] - projected_pt[1])
    if proj_symbol_pt < 0:
        return -1, projected_pt, proj_symbol_pt
    elif proj_symbol_pt > 0:
        return 1, projected_pt, proj_symbol_pt
    else:
        return 0, projected_pt, proj_symbol_pt
    
def categorize_dash_pattern(cc_sizes):
    refined_cc_sizes = [i for i in cc_sizes if i > 5 ]
    if len(refined_cc_sizes) == 0:
        return 'solid'
    avg = sum(refined_cc_sizes) / float(len(refined_cc_sizes))
    if max(refined_cc_sizes) > 200 or len(refined_cc_sizes) == 1:
        return 'solid'
    elif abs(avg - 20) < 10 or len(refined_cc_sizes) >= 4:
        return 'dotted'
    elif abs(avg - 50) < 10 or len(refined_cc_sizes) < 4:
        return 'dash'
    else:
        return 'unknown'

def extract_symbol_along_line(map_name,\
                              shapefile_path,\
                              patch_path = '/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines', \
                              legend_dir = '/data/weiweidu/criticalmaas_data/training',\
                              patch_size=256, roi_buffer=50, match_threshold=0.9):
    
    polylines = read_shp(shapefile_path)
    
    print('number of lines in the original shapefile: ', len(polylines))
  
    all_patches = os.listdir(patch_path)

    line_dict = {}
    for patch_name in all_patches:
        row, col = patch_name.split('.')[0].split('_')[-2:]
        row, col = int(row), int(col)
        bounding_box = box(row, col, row+patch_size, col+patch_size)
        for idx, line in enumerate(polylines):
            line_shp = LineString(line)
            clipped_line = line_shp.intersection(bounding_box)
            if not clipped_line.is_empty:
                empty_image = np.zeros((patch_size, patch_size, 3))
                line_mask = draw_line_in_image(line, empty_image, row, col, buffer=roi_buffer)
                patch_img = cv2.imread(os.path.join(patch_path, patch_name))
                ####################
                patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                ####################
                roi_img = patch_img * line_mask
                
                # binarize the image
                bin_roi_img = binarize_image(roi_img)
             
                # dash pattern detection
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_roi_img, connectivity=8)
                sizes = sorted(stats[:, -1])       
                dash_pattern = categorize_dash_pattern(sizes[:-2])   
                line_dict[str(line_shp)] = [dash_pattern]

    return line_dict
        
        
if __name__ == '__main__':
    map_name = 'NV_HiddenHills'
    extract_symbol_along_line('AK_Dillingham')