'''
cut-off when the height > 1000-pixel
as well as the width
cut above the next symbol bounding box
'''

import os
import cv2
import json
import numpy as np

def get_polygon_bounding_box(polygon_points):
    """
    Calculate the bounding box for a polygon.

    Args:
    - polygon_points: (#points, 2)

    Returns:
    - A tuple representing the bounding box: (min_x, min_y, max_x, max_y)
    """
    min_x = min(polygon_points, key=lambda point: point[0])[0]
    min_y = min(polygon_points, key=lambda point: -point[1])[1]
    max_x = max(polygon_points, key=lambda point: point[0])[0]
    max_y = max(polygon_points, key=lambda point: -point[1])[1]
    return int(min_x), int(-min_y), int(max_x), int(-max_y)

def is_bbox_in_roi(bbox1, bbox2):
    '''
    Check if bbox1 is within bbox2.
    
    bbox1: [y1,x1,y2,x2]
    bbox2 is roi_area: [x,y,w,h]
    '''

    # Convert to top-left and bottom-right
    x1, y1, x2, y2 = bbox1
    x2, y2, w2, h2 = bbox2

    top_left_1 = (x1, y1)
    bottom_right_1 = (x2, y2)

    top_left_2 = (x2, y2)
    bottom_right_2 = (x2 + w2, y2 + h2)

    # Check if bbox1 is within bbox2
    return (top_left_2[0] <= top_left_1[0] <= bottom_right_2[0]) and \
           (top_left_2[1] <= top_left_1[1] <= bottom_right_2[1]) and \
           (top_left_2[0] <= bottom_right_1[0] <= bottom_right_2[0]) and \
           (top_left_2[1] <= bottom_right_1[1] <= bottom_right_2[1])


def read_symbol_bbox_from_json_gt(json_dir, map_name, roi_area=None):
    # the json file is from the competition ground truth
    json_path = os.path.join(json_dir, map_name+'.json')
    json_file = open(json_path)
    metadata = json.load(json_file)
    bbox_list = []
    for symbol in metadata['shapes']:
        bbox = [symbol['points'][0][0], symbol['points'][0][1], symbol['points'][1][0], symbol['points'][1][1]]
        if roi_area and is_bbox_in_roi(bbox, roi_area):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_list.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
        elif roi_area is None:
            bbox_list.append(bbox)
    return bbox_list

def read_symbol_bbox_from_json(json_dir, map_name, roi_area=None):
    # the json file is from the map layout analysis output (Fandel's output)
    bbox_list = []
    json_path = os.path.join(json_dir, map_name+'_PolygonType.geojson')
    with open(json_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    for symbol in metadata['features']:
        polygon = symbol['geometry']['coordinates']
        polygon_np = np.array(polygon[0])
        x1, y1, x2, y2 = get_polygon_bounding_box(polygon_np)
        bbox = [x1, y1, x2, y2]
        if roi_area and is_bbox_in_roi(bbox, roi_area):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_list.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
        elif roi_area is None:
            bbox_list.append(bbox)
            
    json_path = os.path.join(json_dir, map_name+'_PointLineType.geojson')
    with open(json_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    for symbol in metadata['features']:
        polygon = symbol['geometry']['coordinates']
        polygon_np = np.array(polygon[0])
        x1, y1, x2, y2 = get_polygon_bounding_box(polygon_np)
        bbox = [x1, y1, x2, y2]
        if roi_area and is_bbox_in_roi(bbox, roi_area):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bbox_list.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
        elif roi_area is None:
            bbox_list.append(bbox)
    return bbox_list
            

def read_roi_from_json_gt(json_path, map_name):
    # the json file is from the Uncharted ground truth
    with open(json_path, 'r') as file:
        legend_data = json.load(file)
    
    map_id = None
    for item in legend_data['images']:
        if f'{map_name}.tif' in item['file_name']:
            map_id = item['id']
    
    '''
    item['category_id']:
    0: polygon legend area
    1: point and line legend area 
    2: map area
    '''
    
    bboxes = []
    if map_id:
        for item in legend_data['annotations']:
            if item['image_id'] == map_id and item['category_id'] != 2:
                print(item['category_id'])
                bbox = item['bbox']
                bboxes.append(bbox)
                print(f'{map_name} has {bbox}')

    else:
        print(f'{map_name} is not in {json_path}')
        return []
    return bboxes

def read_ln_pt_roi_from_json(json_dir, map_name):
    # the json file is from the Uncharted outputs
    json_path = os.path.join(json_dir, map_name)
    with open(json_path, 'r') as file:
        legend_data = json.load(file)
    
    '''
    'class_label': {map,legend_polygons,legend_points_lines}
    '''
    
    bboxes = []
    for item in legend_data['segments']:
        if item['class_label'] == 'legend_points_lines':
            print(item['bbox'])
            bboxes.append(item['bbox'])
    return bboxes

def read_poly_roi_from_json(json_dir, map_name):
    # the json file is from the Uncharted outputs
    json_path = os.path.join(json_dir, map_name)
    with open(json_path, 'r') as file:
        legend_data = json.load(file)
    
    '''
    'class_label': {map,legend_polygons,legend_points_lines}
    '''
    
    bboxes = []
    for item in legend_data['segments']:
        if item['class_label'] == 'legend_polygons':
            print(item['bbox'])
            bboxes.append(item['bbox'])
    return bboxes


def cutoff_roi_height(symbol_in_roi_dict, MAX_HEIGHT=1000):
    small_roi_list = []

    for roi_bbox, symbol_bbox_list in symbol_in_roi_dict.items():
        roi_x, roi_y, roi_w, roi_h = np.array(roi_bbox.split('_')).astype('int32')

        if roi_h > MAX_HEIGHT and symbol_bbox_list != []:
            bbox_ys = [int(box[1]) for box in symbol_bbox_list]
            sorted_bbox_ys = sorted(bbox_ys)
            count = 1
            s_y = sorted_bbox_ys[0]
            while count < len(sorted_bbox_ys):
                e_y = sorted_bbox_ys[count]
                if e_y - s_y >= MAX_HEIGHT:
                    small_roi_list.append([roi_x, s_y, roi_w, e_y - s_y])
                    s_y = sorted_bbox_ys[count]
                count += 1       
            if e_y != s_y:
                small_roi_list.append([roi_x, s_y, roi_w, roi_y+roi_h-s_y])
        else:
            small_roi_list.append([roi_x, roi_y, roi_w, roi_h])
    return small_roi_list

def cutoff_roi_width(symbol_in_roi_dict, MAX_WIDTH=800, cut_buffer=80):
    small_roi_list = []
    
    for roi_bbox, symbol_bbox_list in symbol_in_roi_dict.items():
        roi_x, roi_y, roi_w, roi_h = np.array(roi_bbox.split('_')).astype('int32')
        
        if roi_w > MAX_WIDTH and symbol_bbox_list != []:
            bbox_xs = [int(box[0]) for box in symbol_bbox_list]
            sorted_bbox_xs = sorted(bbox_xs)
            s_x = sorted_bbox_xs[0]# - cut_buffer
            count = 1
            while count < len(sorted_bbox_xs):
                s_x = min(s_x, sorted_bbox_xs[count])
                e_x = sorted_bbox_xs[count]
                if e_x - s_x >= MAX_WIDTH:
                    small_roi_list.append([s_x, roi_y, e_x - s_x, roi_h])
                    s_x = sorted_bbox_xs[count]# - cut_buffer
                count += 1
            if e_x != s_x:
                print(e_x, s_x, e_x - s_x)
                small_roi_list.append([s_x, roi_y, roi_x+roi_w-s_x, roi_h])
        else:
            small_roi_list.append([roi_x, roi_y, roi_w, roi_h])
    return small_roi_list
            
    

def save_png(map_dir, map_name, symbol_in_roi_dict, output_dir='/data/weiweidu/gpt4/map_legend_gpt_input'):
    map_image_path = os.path.join(map_dir, map_name+'.tif')
    map_image = cv2.imread(map_image_path)
    
    for roi_bbox, symbol_bbox_list in symbol_in_roi_dict.items():
        roi_x, roi_y, roi_w, roi_h = np.array(roi_bbox.split('_')).astype('int32')
        map_roi = map_image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        output_path = os.path.join(output_dir, f"{map_name}_{roi_bbox}.png")
        cv2.imwrite(output_path, map_roi)
        print(f'*** saving {output_path} ***')
    

def main(map_dir, legend_json_path, symbol_json_dir, map_name, output_dir):
    all_symbol_bbox = {'line': [], 'point': [], 'polygon': []}
    
    ##########################################
    # process the line and point legend area
    ln_pt_roi_list = read_ln_pt_roi_from_json(legend_json_path, map_name+'_map_segmentation.json')
    symbol_in_roi = {}
    
    for roi in ln_pt_roi_list:
        bboxes = read_symbol_bbox_from_json_gt(symbol_json_dir, map_name, roi)
        roi_str = '_'.join(str(int(i)) for i in roi)
        symbol_in_roi[roi_str] = bboxes
        # separte point and line features according to 
        # the ratio of width to height of the bounding boxes
        for x1, y1, x2, y2 in bboxes:
            if float(x2-x1)/(y2-y1) < 2.0:
                all_symbol_bbox['point'].append([x1, y1, x2, y2])
            else:
                all_symbol_bbox['line'].append([x1, y1, x2, y2])
    
    save_png(map_dir, map_name, symbol_in_roi, output_dir)
    ##########################################
    
    ##########################################
    # process the polygon legend area
    roi_list = read_poly_roi_from_json(legend_json_path, map_name+'_map_segmentation.json')
    symbol_in_roi = {}
    
    for roi in roi_list:
        print('***', symbol_json_dir, map_name, '***')
        bboxes = read_symbol_bbox_from_json_gt(symbol_json_dir, map_name, roi)
        roi_str = '_'.join(str(int(i)) for i in roi)
        symbol_in_roi[roi_str] = bboxes
        all_symbol_bbox['polygon'].extend(bboxes)
    
    # break the poly legend area into smaller ones for GPT4
    small_rois_width = cutoff_roi_width(symbol_in_roi)
    cutoff_symbol_in_roi_width = {}
    for roi in small_rois_width:
        bboxes = read_symbol_bbox_from_json_gt(symbol_json_dir, map_name, roi)
        roi_str = '_'.join(str(int(i)) for i in roi)
        cutoff_symbol_in_roi_width[roi_str] = bboxes
#     save_png(map_dir, map_name, cutoff_symbol_in_roi_width, output_dir)
    
    small_rois_height = cutoff_roi_height(cutoff_symbol_in_roi_width)    
    cutoff_symbol_in_roi_height = {}
    for roi in small_rois_height:
        bboxes = read_symbol_bbox_from_json_gt(symbol_json_dir, map_name, roi)
        roi_str = '_'.join(str(int(i)) for i in roi)
        cutoff_symbol_in_roi_height[roi_str] = bboxes
    
    save_png(map_dir, map_name, cutoff_symbol_in_roi_height, output_dir)
    
    return all_symbol_bbox

    
if __name__ == '__main__':
    symbol_json_dir = '/data/weiweidu/criticalmaas_data/validation'
    map_dir = '/data/weiweidu/criticalmaas_data/validation'
#     symbol_json_dir = '/data/weiweidu/criticalmaas_data/training'
#     map_dir = '/data/weiweidu/criticalmaas_data/training'
    map_name = 'DC_Frederick'#'DC_Wash_West'#'MN' #
    
    legend_json_path = '/data/weiweidu/map_legend_segmentation_labels/ch2_validation_evaluation_labels_coco.json'
#     legend_json_path = '/data/weiweidu/map_legend_segmentation_labels/ch2_training_labels_coco.json'
    
    output_dir = '/data/weiweidu/gpt4/map_legend_gpt_input_hw'
    symbol_bbox_in_roi = main(map_dir, legend_json_path, symbol_json_dir, map_name, output_dir)
#     print(symbol_bbox_in_roi)