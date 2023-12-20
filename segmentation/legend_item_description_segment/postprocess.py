import json
import os
import numpy as np
from gpt4_input_generation import get_polygon_bounding_box

def combine_json_files_from_gpt(json_dir, map_name, output_path):
    combined_data = {}
    for root, dirs, files in os.walk(json_dir):
        for f_name in files:
            if map_name not in f_name: 
                continue
            json_path = os.path.join(root, f_name)
            
            with open(json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            combined_data = {**combined_data, **data}
    
    with open(output_path, "w") as outfile:
        json.dump(combined_data, outfile)
    return 

def combine_json_files_gpt_and_symbol_legend(json_dir_symbol, json_dir_legend, gpt_dir, map_name, output_path):    
    json_gpt_path = os.path.join(gpt_dir, map_name+'.json')
    
    json_poly_sym_path = os.path.join(json_dir_symbol, map_name+'_PolygonType.geojson')
    json_legend_path = os.path.join(json_dir_legend, map_name+'_map_segmentation.json')
    
    with open(json_gpt_path, 'r', encoding='utf-8') as file:
        gpt_dict = json.load(file)

    with open(json_poly_sym_path, 'r', encoding='utf-8') as file:
        sym_poly_dict = json.load(file)
        
    with open(json_legend_path, 'r', encoding='utf-8') as file:
        legend_dict = json.load(file)
        
    for item in legend_dict['segments']:
        if item['class_label'] == 'legend_polygons':
            poly_bbox = item['bbox']
        if item['class_label'] == 'legend_points_lines':
            ptln_bbox = item['bbox']
    
    comb_dict = {}

    for symbol in sym_poly_dict['features']:
        polygon = symbol['geometry']['coordinates']
        polygon_np = np.array(polygon[0])
        x1, y1, x2, y2 = get_polygon_bounding_box(polygon_np)
        bbox = [x1, y1, x2, y2]
        if str(bbox) not in gpt_dict.keys():
            continue
        if is_bbox_in_roi(bbox, poly_bbox):
            comb_dict[str(bbox)] = {**gpt_dict[str(bbox)], \
                                    **{'geometry': symbol['geometry']['coordinates']}, \
                                    **{'properties': symbol['properties']},\
                                    **{'legend_class': 'polygon'}}
        elif is_bbox_in_roi(bbox, ptln_bbox):
            comb_dict[str(bbox)] = {**gpt_dict[str(bbox)], \
                                    **{'geometry': symbol['geometry']['coordinates']}, \
                                    **{'properties': symbol['properties']},\
                                    **{'legend_class': 'point_line'}}
        
    for bbox, val in gpt_dict.items():
        if bbox not in comb_dict.keys():
            if is_bbox_in_roi(eval(bbox), poly_bbox):
                comb_dict[bbox] = {**val, **{'legend_class': 'polygon'}}
            elif is_bbox_in_roi(eval(bbox), ptln_bbox):
                comb_dict[bbox] = {**val, **{'legend_class': 'point_line'}}
    
    with open(output_path, "w") as outfile:
        json.dump(comb_dict, outfile)
    
    print(f'*** saved result in {output_path} ***')
    return 

