import json
import os
import cv2
import numpy as np
from gpt4_input_generation import get_polygon_bounding_box

def is_bbox_in_roi(bbox, roi_bbox):
    x, y, w, h = roi_bbox
    x1, y1, x2, y2 = bbox

    if x1 >= x and y1 >= y and x2 <= x+w and y2 <= y+h:
        return True
    return False

def combine_json_files_from_gpt(json_dir, map_name, output_path, category):
    combined_data = {}
    for root, dirs, files in os.walk(json_dir):
        for f_name in files:
            if map_name not in f_name or category not in f_name.split('_')[-1]: 
                continue
            json_path = os.path.join(root, f_name)
            print(json_path)
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            combined_data = {**combined_data, **data}
            print(combined_data)
    with open(output_path, "w") as outfile:
        json.dump(combined_data, outfile)
    print(f'*** saved result in {output_path} ***')
    return 

def combine_json_files_gpt_and_symbol(json_dir_symbol, gpt_dir, map_name, output_path, category):    
    json_gpt_path = os.path.join(gpt_dir, map_name+f'_{category}.json')
    
#     json_poly_sym_path = os.path.join(json_dir_symbol, map_name+'_PolygonType.geojson')
# #     json_lnpt_sym_path = os.path.join(json_dir_symbol, map_name+'_PointLineType.geojson')
    
#     with open(json_gpt_path, 'r', encoding='utf-8') as file:
#         gpt_dict = json.load(file)

#     with open(json_poly_sym_path, 'r', encoding='utf-8') as file:
#         sym_poly_dict = json.load(file)
        
#     comb_dict = {}

#     for symbol in sym_poly_dict['features']:
#         polygon = symbol['geometry']['coordinates']
#         polygon_np = np.array(polygon[0])
#         x1, y1, x2, y2 = get_polygon_bounding_box(polygon_np)
#         bbox = [x1, y1, x2, y2]
#         if str(bbox) not in gpt_dict.keys():
#             continue
#         comb_dict[str(bbox)] = {**gpt_dict[str(bbox)], \
#                                 **{'geometry': symbol['geometry']['coordinates']}, \
#                                 **{'properties': symbol['properties']}}
        
    for bbox, val in gpt_dict.items():
        comb_dict[bbox] = val
    
    with open(output_path, "w") as outfile:
        json.dump(comb_dict, outfile)
    
    print(f'*** saved result in {output_path} ***')
    return 



def combine_json_files_gpt_and_symbol_legend(json_dir_symbol, json_dir_legend, gpt_dir, map_dir, map_name,\
                                             output_path, category):    
    
    json_gpt_path = os.path.join(gpt_dir, map_name+f'_{category}_.json')
    
#     json_poly_sym_path = os.path.join(json_dir_symbol, map_name+'_PolygonType.geojson')
    json_legend_path = os.path.join(json_dir_legend, map_name+'_map_segmentation.json')
    
    with open(json_gpt_path, 'r', encoding='utf-8') as file:
        gpt_dict = json.load(file)

#     with open(json_poly_sym_path, 'r', encoding='utf-8') as file:
#         sym_poly_dict = json.load(file)
        
    with open(json_legend_path, 'r', encoding='utf-8') as file:
        legend_dict = json.load(file)
        
    for item in legend_dict['segments']:
        if item['class_label'] == 'legend_polygons':
            poly_bbox = item['bbox']
        if item['class_label'] == 'legend_points_lines':
            ptln_bbox = item['bbox']
        if item['class_label'] == 'map':
            map_content_bbox = item['bbox'] 
    
#     comb_dict = {}

#     for symbol in sym_poly_dict['features']:
#         polygon = symbol['geometry']['coordinates']
#         polygon_np = np.array(polygon[0])
#         x1, y1, x2, y2 = get_polygon_bounding_box(polygon_np)
#         bbox = [x1, y1, x2, y2]
#         if str(bbox) not in gpt_dict.keys():
#             continue
#         if is_bbox_in_roi(bbox, poly_bbox):
#             comb_dict[str(bbox)] = {**gpt_dict[str(bbox)], \
#                                     **{'geometry': symbol['geometry']['coordinates']}, \
#                                     **{'properties': symbol['properties']},\
#                                     **{'legend_class': 'polygon'}}
#         elif is_bbox_in_roi(bbox, ptln_bbox):
#             comb_dict[str(bbox)] = {**gpt_dict[str(bbox)], \
#                                     **{'geometry': symbol['geometry']['coordinates']}, \
#                                     **{'properties': symbol['properties']},\
#                                     **{'legend_class': 'point_line'}}

#     for bbox, val in gpt_dict.items():
#         if bbox not in comb_dict.keys():
#             if is_bbox_in_roi(eval(bbox), poly_bbox):
#                 comb_dict[bbox] = {**val, **{'legend_class': 'polygon'}}
#             elif is_bbox_in_roi(eval(bbox), ptln_bbox):
#                 comb_dict[bbox] = {**val, **{'legend_class': 'point_line'}}
    
    gpt_dict['map_content_box'] = map_content_bbox
    
    map_img = cv2.imread(os.path.join(map_dir, map_name+'.tif'))
    img_h, img_w = map_img.shape[:-1]
    gpt_dict['map_dimension'] = [img_h, img_w]
    
    with open(output_path, "w") as outfile:
        json.dump(gpt_dict, outfile)
    
    print(f'*** saved result in {output_path} ***')
    return 

