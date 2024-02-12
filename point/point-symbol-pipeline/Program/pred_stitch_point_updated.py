import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from PIL import ImageFile
from PIL import Image 
import os
import json
from ultralytics import YOLO
import glob
import pandas as pd 
import argparse
from geojson import Polygon, Feature, FeatureCollection, dump,Point
import geojson
import logging
# import pdb
import geopandas
# logging.basicConfig(level=logging.INFO) 


def predict_img_patches(crop_dir_path,model_dir_root,selected_models,predict_output_dir):
    # for dir_per_map in os.listdir(crop_dir_path):
    #     dir_per_map_path=os.path.join(crop_dir_path,dir_per_map)
    #     print(dir_per_map_path)
    # for img_dir in os.listdir(crop_dir_path):          
    #     val_data_path=os.path.join(crop_dir_path,img_dir)
    #     print(val_data_path)
    map_name = os.path.basename(os.path.dirname(crop_dir_path))
    output_path=os.path.join(predict_output_dir,map_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)     
    print(output_path)
    for val_img in os.listdir(crop_dir_path):            
        if val_img.endswith('.jpg') or val_img.endswith('.png'):
            entire_res=[] 
            img_path=os.path.join(crop_dir_path,val_img)  
            for idx,model_file in enumerate(selected_models):
                weight_path=os.path.join(model_dir_root,model_file)
                print(weight_path)
                model = YOLO(weight_path)
                pnt_name=model_file.split('.')[0]                      
                results = model(img_path,conf=0.25)  # results list
                res_boxes = results[0].boxes.data.cpu().numpy()             
                for i, box in enumerate(res_boxes):
                    res_per_crop={}
                    res_per_crop['img_geometry']=[]
                    res_per_crop['type']=None
                    # res_per_crop['score']=[]
                    x1, y1, x2, y2, conf, _ = box
                    cnt_x=int((x1+x2)/2)
                    cnt_y=int((y1+y2)/2)
                    plt.scatter(cnt_x, cnt_y,s=1)
                    res_per_crop['img_geometry'].append([cnt_x,cnt_y])
                    res_per_crop['type']=pnt_name
                    entire_res.append(res_per_crop)                   
            
            out_file_path=os.path.join(output_path,val_img.split('.')[0]+'.json')
            with open(out_file_path, "w") as out:
                json.dump(entire_res, out)

def stitch_to_single_result(crop_dir_path,pred_root,stitch_root,crop_shift_size=1000): 
    map_name = os.path.basename(os.path.dirname(crop_dir_path))
    shift_size = crop_shift_size
    file_list = glob.glob(os.path.join(pred_root,map_name) + '/*.json')
    output_geojson = os.path.join(stitch_root,map_name+'.geojson')
    file_list = sorted(file_list)
    # if len(file_list) == 0:
    #     logging.warning('No files found for %s' % map_subdir)       
    map_data = []
    for file_path in file_list:
        get_h_w = os.path.basename(file_path).split('.')[0].split('_')
        patch_index_h = int(get_h_w[-2][1:])
        patch_index_w = int(get_h_w[-1][1:])
        try:
            df = pd.read_json(file_path, dtype={"type":object})
        except pd.errors.EmptyDataError:
            logging.warning('%s is empty. Skipping.' % file_path)
            continue 
        except KeyError as ke:
            logging.warning('%s has no detected labels. Skipping.' %file_path)
            continue         
        for index, line_data in df.iterrows():
            # print(line_data["img_geometry"][0])
            line_data["img_geometry"][0][0] = line_data["img_geometry"][0][0] + shift_size * patch_index_w
            line_data["img_geometry"][0][1] = line_data["img_geometry"][0][1] + shift_size * patch_index_h
            # print('after',line_data["img_geometry"][0])
        map_data.append(df)     
    map_df = pd.concat(map_data)
    idx=0
    features = []
    for index, line_data in map_df.iterrows():
        img_x=line_data['img_geometry'][0][0]
        img_y=line_data['img_geometry'][0][1]
        point= Point([img_x,img_y])
        sym_type = line_data['type']
        idx+=1
        features.append(Feature(geometry = point, properties={'type': sym_type, "id": idx} ))
    feature_collection = FeatureCollection(features)
    with open(output_geojson, 'w', encoding='utf8') as f:
        dump(feature_collection, f, ensure_ascii=False)

def list_of_strings(arg):
    return arg.split(',')
def main(args):
    input_dir_root=args.input_dir_root
    model_weights_dir=args.model_weights_dir
    selected_model_weights=args.selected_model_weights
    output_dir_root=args.output_dir_root

    predict_output_dir=os.path.join(output_dir_root,'prediction')
    if not os.path.isdir(predict_output_dir):
            os.mkdir(predict_output_dir)

    print("=== Running a model prediction module ===")
    predict_img_patches(input_dir_root,model_weights_dir,selected_model_weights,predict_output_dir)
       
    stitch_output_dir=os.path.join(output_dir_root,'stitch')
    if not os.path.isdir(stitch_output_dir):
            os.mkdir(stitch_output_dir)

    print("=== Running a stitching module ===")
    stitch_to_single_result(input_dir_root,predict_output_dir,stitch_output_dir,crop_shift_size=1000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir_root', type=str, 
                        help='input image patches directory per map')
    parser.add_argument('--model_weights_dir', type=str,
                        help='directory contains entire model weights for prediction')
    parser.add_argument('--selected_model_weights', type=list_of_strings,
                        help='selected model weights for prediction')
    parser.add_argument('--output_dir_root', type=str,
                        help='output directory for saving intermediate results')                     
    args = parser.parse_args()
    main(args)

#example command line
#python pred_stitch_point.py --input_dir_root /home/yaoyi/jang0124/mapkurator/output/pipeline_sample/crop/OR_Carlton/ --model_weights_dir /home/yaoyi/jang0124/critical-maas/code/point-symbol/release/model/model_weights/ --selected_model_weights 'gravel_pit.pt','drill_hole.pt' --output_dir_root /home/yaoyi/jang0124/critical-maas/code/pipeline/sample/
