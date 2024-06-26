import json
import os
import numpy as np
import pytesseract
from PIL import Image

def get_polygon_bounding_box(polygon_points):
    """
    Calculate the bounding box for a polygon.

    Args:
    - polygon_points: (#points, 2)

    Returns:
    - A tuple representing the bounding box: (min_x, min_y, max_x, max_y)
    """
    pts_np = np.array(polygon_points).astype('int32')
    min_x, max_x = np.min(polygon_points[:,0]), np.max(polygon_points[:,0])
    min_y, max_y = np.min(polygon_points[:,1]), np.max(polygon_points[:,1])
#     min_x = min(polygon_points, key=lambda point: point[0])[0]
#     min_y = min(polygon_points, key=lambda point: point[1])[1]
#     max_x = max(polygon_points, key=lambda point: point[0])[0]
#     max_y = max(polygon_points, key=lambda point: point[1])[1]
    return int(min_x), int(min_y), int(max_x), int(max_y)

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
#                 print(item['category_id'])
                bbox = item['bbox']
                bboxes.append(bbox)
#                 print(f'{map_name} has {bbox}')

    else:
        print(f'{map_name} is not in {json_path}')
        return []
    return bboxes

def read_roi_from_json(json_dir, map_name):
    # the json file is from the Uncharted outputs
    json_path = os.path.join(json_dir, map_name)
    with open(json_path, 'r') as file:
        legend_data = json.load(file)    
    '''
    'class_label': {map,legend_polygons,legend_points_lines}
    '''
    bboxes = []
    for item in legend_data['segments']:
        if item['class_label'] == 'map':
            continue
        bboxes.append(item['bbox'])
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
            bboxes.append(item['bbox'])
    return bboxes

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
    '''
    the json file is from the legend-item output (Fandel's output)
    the bbox is [col_min, row_min, col_max, row_max]
    '''
    
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

def choose_symbol_bbox_in_roi(all_symbol_bbox, roi_area=None):
    bboxes_in_roi  = []
    for bbox in all_symbol_bbox:
        if roi_area and is_bbox_in_roi(bbox, roi_area):
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            bboxes_in_roi.append([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)])
    return bboxes_in_roi

def bboxes2points(bbox_list):
    """
    convert a list of bboxes to 
    a list of center points of bbox
    """
    points = []
    for bbox in bbox_list:
        x1, y1, x2, y2 = bbox # order by (col_min,row_min, col_max, row_max)
        c_x, c_y = (x1+x2)/2.0, (y1+y2)/2.0
        points.append([c_x, c_y])
#     points.sort(key=lambda x:(x[1], x[0]))
    return points

def ocr_bbox(bbox_img):   
    ocr = pytesseract.image_to_data(bbox_img, output_type=pytesseract.Output.DICT)
#     print(ocr['text'])
    res = ""
    for i in range(len(ocr['text'])):
        if int(ocr['conf'][i]) > 0.1 and ocr['text'][i] != '' and not ocr['text'][i].isspace():  # Confidence threshold.
            text = ocr['text'][i]#.encode('ascii', 'ignore').decode('ascii')
            res += ' ' + text
#             if not text.isspace() or text == '':
#                 return text
    return res

def read_legend_json(legend_json_dir, map_name):
    legend_json_path = os.path.join(legend_json_dir, map_name+'_map_segmentation.json')
    with open(legend_json_path, 'r', encoding='utf-8') as file:
        legend_dict = json.load(file)
    map_content_bbox, poly_bbox, ptln_bbox = None, None, None    
    for item in legend_dict['segments']:
        if item['class_label'] == 'legend_polygons':
            poly_bbox = item['bbox']
        if item['class_label'] == 'legend_points_lines':
            ptln_bbox = item['bbox']
        if item['class_label'] == 'map':
            map_content_bbox = item['bbox'] 
    return map_content_bbox, poly_bbox, ptln_bbox