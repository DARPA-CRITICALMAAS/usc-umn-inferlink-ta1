'''
The bboxes are grouped by columns in input_process.py
This script crops one image for one column 
the cropped images are inputs for GPT4
'''
import os
import cv2
import json
import numpy as np
from PIL import Image
import time
import collections
import io

def get_image_size(image):
    img_pil = Image.fromarray(image)
    # Save the image to an in-memory bytes buffer
    img_buffer = io.BytesIO()
    img_pil.save(img_buffer, format='PNG')
    # Calculate the size in MB
    img_size_mb = img_buffer.getbuffer().nbytes / (1024 * 1024)
    return img_size_mb
    
def crop_image_further(bbox_list, s_col, width, height):
    gpt_input_image_bboxes = []
    bbox_list = list(bbox_list)
    bbox_list.sort(key=lambda x:x[1]) # sorted by row
    num_bbox = len(bbox_list)
    # divide the height into three parts
    row_ranges = np.linspace(0, num_bbox-1, num=4).astype('int32')
    for i in range(1, row_ranges.shape[0]):
        s_row, e_row = bbox_list[row_ranges[i-1]][1], bbox_list[row_ranges[i]][1]
        e_col = s_col + width
        gpt_input_image_bboxes.append([s_col, e_col, s_row, e_row])
    return gpt_input_image_bboxes
    

def generate_gpt4_input(column_bboxes, roi, map_tif, map_name, save_gpt_input_dir):
    x, y, w, h = np.array(roi).astype('int32')
    columns_png = collections.defaultdict(list)
    
    if not os.path.exists(save_gpt_input_dir):
        os.mkdir(save_gpt_input_dir)
    
    if column_bboxes == {}:
        img_name = f'{map_name}_{x}_{y}_{w}_{h}.png'
        gpt_input_png_path = os.path.join(save_gpt_input_dir, img_name)
        gpt_input_png = map_tif[y:y+h, x:x+w]
        cv2.imwrite(gpt_input_png_path, gpt_input_png)
        columns_png['0.0'] = [img_name]
        return columns_png
    
    sorted_columns = sorted(column_bboxes.keys())
    
    for ind, col in enumerate(sorted_columns):
        bbox_np = np.array(column_bboxes[col]).astype('int32')

        s_col = np.min(bbox_np[:, 0])
        
        if ind < len(sorted_columns)-1 and sorted_columns[ind+1]-s_col > 1000:
            e_col = sorted_columns[ind+1]
        else:    
            e_col = x + int(w)
        
        e_col = int(e_col)
        
        s_row = np.min(bbox_np[:, 1])
        e_row = int(s_row+h)

        if abs(s_row-e_row) < 10 or abs(s_col-e_col) < 10:
            continue
        
        gpt_input_png = map_tif[s_row:e_row, s_col:e_col]
        
        if get_image_size(gpt_input_png) > 18: #image size > 18MB
            gpt_input_bbox = crop_image_further(bbox_np, s_col, int(e_col)-int(s_col), int(e_row)-int(s_row))
        else:
            gpt_input_bbox = [[s_col, e_col, s_row, e_row]]
        
        # columns_png save the gpt input png name for one column
        for s_col, e_col, s_row, e_row in gpt_input_bbox:
            png_name = f'{map_name}_{int(s_col)}_{int(s_row)}_{int(e_col)-int(s_col)}_{int(e_row)-int(s_row)}.png'
            columns_png[col].append(png_name)
            
            gpt_input_png_path = os.path.join(save_gpt_input_dir, png_name)
            gpt_input_png = map_tif[s_row:e_row, s_col:e_col]
            cv2.imwrite(gpt_input_png_path, gpt_input_png)
    return columns_png
        
    
if __name__ == '__main__':
    legend_json_path = '/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps_legend_outputs'
    symbol_json_dir = '/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps_legend_item_outputs'
    map_dir = '/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps'
    # testing maps: 10705_61989, 1188_23017, 17_10042, 70_13052, 21257_13870, 262_9137, 54565_18569, 67434_112084, 15539_43379
    map_name = '1188_23017'
    
    output_dir = 'gpt_inputs'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    generate_gpt_input(map_dir, legend_json_path, symbol_json_dir, map_name, output_dir)
    