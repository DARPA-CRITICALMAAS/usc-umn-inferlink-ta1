import os
import cv2
import numpy as np
from helper.process_shp import read_shp
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from PIL import Image
import math
import json
from sklearn.cluster import KMeans

def merge_lines(line_list, line_to_merge):
    if line_list is None:
        return [LineString(line_to_merge)]
    for i, line in enumerate(line_list):
        if line_to_merge.intersects(line):
            merged_line = unary_union([line, line_to_merge])
            del line_list[i]
            line_list.append(merged_line)
            return line_list
    line_list.append(line_to_merge)
    return line_list

def draw_line_in_image(line_list, empty_image, row, col, buffer=50):
    for i in range(1, len(line_list)):
        pt1 = (int(line_list[i-1][1])-col, int(line_list[i-1][0])-row)
        pt2 = (int(line_list[i][1])-col, int(line_list[i][0])-row)
        cv2.line(empty_image, pt1, pt2, 1, buffer)
    return empty_image

def binarize_image(line_area_bin_img):
    avg_gray_val = np.average(line_area_bin_img[np.where(line_area_bin_img>0)])
    min_gray_val = np.min(line_area_bin_img)
    binary_thres = (avg_gray_val + min_gray_val) // 2
    _, binary_image = cv2.threshold(line_area_bin_img, binary_thres, 255, cv2.THRESH_BINARY_INV)
    return binary_image.astype('uint8')

def remove_small_outliers(data):
    data = np.array(data)
    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Determine the outlier cutoffs
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
#     outliers = data[(data < lower_bound) | (data > upper_bound)]
    outliers = data[(data < lower_bound)]
    refined_data = [i for i in data if i not in outliers]
    return refined_data

def cluster_connected_components(data):
    data = np.array(data).reshape((-1,1))
    # Initialize KMeans with 2 clusters
    initial_centroids = [[50], [300]]
    kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1)
    # Fit KMeans on the data
    kmeans.fit(data)
    # Get the cluster assignments (0 or 1 in this case)
    clusters = kmeans.labels_

    # To separate the original data into two clusters
    cluster_1 = data[clusters == 0]
    cluster_2 = data[clusters == 1]
    return cluster_1.flatten(), cluster_2.flatten()
    
def categorize_dash_pattern(cc_sizes):
    refined_cc_sizes = [i for i in cc_sizes if i > 2 ]
    if len(refined_cc_sizes) == 0:
        return 'solid'
    refined_cc_sizes = remove_small_outliers(refined_cc_sizes)    
    if len(refined_cc_sizes) == 1: # 500 threshold is to measure the length
        return 'solid'
    c1_list, c2_list = cluster_connected_components(refined_cc_sizes)
    c1_cent, c2_cent = np.mean(c1_list), np.mean(c2_list)

    chosen_list = c1_list if len(c1_list) > len(c2_list) else c2_list
    chosen_cent = c1_cent if len(c1_list) > len(c2_list) else c2_cent
    if len(c2_list) < 3 and len(c1_list) < 3:
        return 'solid'
    elif 15 < chosen_cent < 100: # the length of dashed lines (10, 100) 
        return 'dashed'   
    elif 0 < chosen_cent <= 15: # the length of dotted lines (0, 10] 
        return 'dotted'
    else:
        return 'solid'

def extract_attributes_along_line(map_name,\
                              shapefile_path,\
                              patch_path = '/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines/', \
                              legend_dir = './symbol_template',\
                              patch_size=256, roi_buffer=30, match_threshold=0.7):
    
    polylines = read_shp(shapefile_path)
    
    print('number of lines in the original shapefile: ', len(polylines))
  
    all_patches = os.listdir(patch_path)

    line_dict = {}
    pattern_dict = {'solid': [], 'dotted': [], 'dashed': [], 'unknown': []}
    for patch_name in all_patches:
        row, col = patch_name.split('.')[0].split('_')[-2:]
        row, col = int(row), int(col)
        bounding_box = box(row, col, row+patch_size, col+patch_size)
        
        for idx, line in enumerate(polylines):
            line_shp = LineString(line)
            clipped_line = line_shp.intersection(bounding_box)
            if not clipped_line.is_empty and (str(line_shp) not in line_dict.keys() or len(line_dict[str(line_shp)])==1):
                ########################################
                # dash pattern detection
                empty_image = np.zeros((patch_size, patch_size))
                line_mask = draw_line_in_image(line, empty_image, row, col, buffer=roi_buffer)
                
                patch_img = cv2.imread(os.path.join(patch_path, patch_name), 0)
                patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                roi_img = patch_img * line_mask   
                # binarize the image
                bin_roi_img = binarize_image(roi_img)
                bin_roi_img = (bin_roi_img * line_mask).astype('uint8')
                thinned_image = cv2.ximgproc.thinning(bin_roi_img, thinningType=cv2.ximgproc.THINNING_GUOHALL)
#                 edges = cv2.Canny(bin_roi_img, 50, 150, apertureSize=3)
#                 lines = cv2.HoughLinesP(thinned_image, 1, np.pi / 180, threshold=100, minLineLength=10, maxLineGap=10)

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thinned_image, connectivity=8)
                sizes = sorted(stats[:, -1])  
                if sizes[:-1] == []:
                    continue
                dash_pattern = categorize_dash_pattern(sizes[:-1])   
                line_dict[str(line_shp)] = [dash_pattern]
                pattern_dict[dash_pattern] = merge_lines(pattern_dict[dash_pattern], line_shp)
               
                ########################################

    return pattern_dict
        
        
if __name__ == '__main__':
    map_name = 'AK_Christian'
    shp_path = f'/data/weiweidu/temp/{map_name}_fault_line_pred.shp'
        
    line_by_category = extract_attributes_along_line(map_name,\
                             f'/data/weiweidu/criticalmaas_data/training_fault_line_comb/{map_name}_fault_line.shp', \
                             patch_path=f'/data/weiweidu/criticalmaas_data/training_cropped_images/{map_name}_g256_s256')
#     write_shp_in_imgcoord_with_attr(shp_path, line_by_category, legend_text=None, feature_name='fault', image_coords=True)
    