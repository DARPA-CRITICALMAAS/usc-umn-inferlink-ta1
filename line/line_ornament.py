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
        cv2.line(empty_image, pt1, pt2, 1, buffer)
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

def translate_augment(image, shifting_list):
    rows, cols = image.shape
    img_aug = []
    for h in shifting_list:
        for v in shifting_list:
            M = np.float32([[1,0,h],[0,1,v]])
            dst = cv2.warpAffine(image, M, (cols,rows), borderMode=cv2.INTER_AREA)
            img_aug.append(dst)
    return img_aug

def template_match_with_aug(main_image, raw_template, threshold=0.7, shift_list=[-10, 5, 0, 5, 10]):
    # Get the width and height of the template
    w, h = raw_template.shape
    w_m, h_m = main_image.shape
    
    if w_m < w or h_m <h:
        print('template is larger', (w, h), (w_m, h_m))
        return False, None
    
    aug_templates = translate_augment(raw_template, shift_list)
    # Perform template matching
    for i, template in enumerate(aug_templates):
        result = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        if loc[0].shape[0] == 0:
            continue
        main_image_copy = main_image.copy()
        for pt in zip(*loc[::-1]):
            cv2.rectangle(main_image_copy, pt, (pt[0] + w, pt[1] + h), 255, 2)
            return True, main_image_copy
    return False, None

def read_images_from_folder(image_dir):
    samples = []
    sample_names = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path, 0)
        samples.append(image)
        sample_names.append(image_name)
    samples = np.array(samples)
    return samples, sample_names
    
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
                              patch_path = '/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines/', \
                              legend_dir = './symbol_template',\
                              patch_size=256, roi_buffer=50, match_threshold=0.7):
    
    polylines = read_shp(shapefile_path)
    
    print('number of lines in the original shapefile: ', len(polylines))
  
    all_patches = os.listdir(patch_path)
    
    pos_images, pos_image_names = read_images_from_folder(legend_dir)

    line_dict = {}
    for patch_name in all_patches:
        row, col = patch_name.split('.')[0].split('_')[-3:-1]
        row, col = int(row), int(col)
        bounding_box = box(row, col, row+patch_size, col+patch_size)
        print(f'*** processing {patch_name} ***')
        for idx, line in enumerate(polylines):
            line_shp = LineString(line)
            clipped_line = line_shp.intersection(bounding_box)
            if not clipped_line.is_empty:
                ########################################
                # dash pattern detection
                empty_image = np.zeros((patch_size, patch_size, 3))
                line_mask = draw_line_in_image(line, empty_image, row, col, buffer=roi_buffer)
                
                patch_img = cv2.imread(os.path.join(patch_path, patch_name))
                patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                roi_img = patch_img * line_mask                
                # binarize the image
                bin_roi_img = binarize_image(roi_img)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_roi_img, connectivity=8)
                sizes = sorted(stats[:, -1])       
                dash_pattern = categorize_dash_pattern(sizes[:-2])   
                line_dict[str(line_shp)] = [dash_pattern]
                ########################################
               
                ########################################
                # line direction detection
                raw_patch = cv2.imread(os.path.join(patch_path, patch_name), 0)
                empty_image = np.zeros(raw_patch.shape)
                draw_line_in_image(line, empty_image, row, col, buffer=80)
                patch_roi = (raw_patch * empty_image).astype('uint8')
                
                for pos_ind in range(pos_images.shape[0]):
                    is_matched, res_img = template_match_with_aug(patch_roi, pos_images[pos_ind], threshold=match_threshold)
                    if is_matched:
                        angle = int(pos_image_names[pos_ind].split('.')[0].split('_')[1])
                        direction = 360 - angle
                        line_dict[str(line_shp)].append(direction)
                        break
                
    return line_dict
        
        
if __name__ == '__main__':
    map_name = 'NV_HiddenHills'
    
    extract_symbol_along_line('NV_HiddenHills',\
                              '/data/weiweidu/LDTR_criticalmaas_test/pred4shp/NV_HiddenHills_fault_line_pred.shp',\
                            patch_path = '/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines/NV_HiddenHills_g256_s100/raw')