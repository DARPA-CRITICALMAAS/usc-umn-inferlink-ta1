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
import torch
import math

def predict_img_patches_strike(map_name,crop_dir_path,model_dir_root,selected_models,predict_output_dir):

    output_path=os.path.join(predict_output_dir,map_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)     
    for val_img in os.listdir(crop_dir_path):          
        entire_res=[] 
        img_path=os.path.join(crop_dir_path,val_img)  
        for idx,model_file in enumerate(selected_models):
            weight_path=os.path.join(model_dir_root,model_file)
            model = YOLO(weight_path)
            pnt_name=model_file.split('.')[0]                 
            results = model(img_path,conf=0.25)  # results list
            res_boxes = results[0]          
            for i, box in enumerate(res_boxes):
                res_per_crop={}
                res_per_crop['img_geometry']=[]
                res_per_crop['score']=[]
                res_per_crop['r_bbox']=[]
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = box.obb.xyxyxyxy.tolist()[0]
                [_, _, _, _, angle] = box.obb.xywhr.tolist()[0]
                angle = math.degrees(angle)
                conf = box.obb.conf.tolist()[0]
                
                cnt_x=int((x1+x3)/2)
                cnt_y=int((y1+y3)/2)
                res_per_crop['img_geometry'].append([cnt_x,cnt_y])
                res_per_crop['type']='strike'
                res_per_crop['score']=str(conf)
                res_per_crop['r_bbox'].append([int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)])
                res_per_crop['strike_angle'] = (int)(angle) + 90
                entire_res.append(res_per_crop)                   
            out_file_path=os.path.join(output_path,val_img.split('.')[0]+'.json')
            with open(out_file_path, "w") as out:
                json.dump(entire_res, out)

def stitch_to_each_strike(map_name, crop_dir_path,pred_root,stitch_root ): 
    # map_name = os.path.basename(os.path.dirname(crop_dir_path))
    # shift_size = crop_shift_size
    file_list = glob.glob(os.path.join(pred_root,map_name) + '/*.json')

    if len(file_list) != 0:
        map_data = []
        for file_path in file_list:
            get_h_w = os.path.basename(file_path).split('.')[0].split('_')
            patch_index_h = int(get_h_w[-2])
            patch_index_w = int(get_h_w[-1])
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
                line_data["img_geometry"][0][0] = line_data["img_geometry"][0][0] + patch_index_w
                line_data["img_geometry"][0][1] = line_data["img_geometry"][0][1] + patch_index_h
                new_bbox = [line_data['r_bbox'][0][0]+patch_index_w, line_data['r_bbox'][0][1]+ patch_index_h, line_data['r_bbox'][0][2] + patch_index_w, line_data['r_bbox'][0][3]+patch_index_h]
                line_data['r_bbox'][0] = new_bbox
                # print('after',line_data["img_geometry"][0])
            map_data.append(df)     
        map_df = pd.concat(map_data)
        idx=0
        features_per_symbol = {}
        for index, line_data in map_df.iterrows():
            img_x = line_data['img_geometry'][0][0]
            img_y = line_data['img_geometry'][0][1]
            point = Point([img_x,img_y])
            sym_type = line_data['type']          
            strike_angle = line_data['strike_angle']
            #following the convention of  dd = st + 90 (degree)
            dip_direction = strike_angle + 90
            sym_type = sym_type.split('.')[0]
            score = line_data['score']
            if len(line_data['r_bbox'][0]) == 4:
                x1,y1,x2,y2 = line_data['r_bbox'][0]
                bbox=[x1,y1,x2,y2]
            else:
                bbox=[0,0,0,0]
            if sym_type not in features_per_symbol.keys():
                features_per_symbol[sym_type] = {"id":str(0),
                                                "crs" : "CRITICALMAAS:pixel",
                                                "cdr_projection_id": None,
                                                "name": sym_type,
                                                "description": None,
                                                "legend_bbox": None, "point_features": list([])}                               
            features_per_symbol[sym_type]["point_features"].append(FeatureCollection([Feature(geometry = point, id= str(len(features_per_symbol[sym_type]["point_features"])), properties={"model":sym_type+'.pt', "confidence": score, "model_version": "v1", "bbox": bbox ,"dip" : None ,"dip_direction" : dip_direction })]))

        stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
        if not os.path.exists(stitch_output_dir_per_map):
            os.makedirs(stitch_output_dir_per_map) 
        if len(features_per_symbol) != 0:
            for each_pnt in features_per_symbol.keys():
                each_file_per_pnt_name=map_name+'_'+each_pnt+'.geojson'
                output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,each_file_per_pnt_name)
                # features_per_symbol[each_pnt]["point_features"] = FeatureCollection(features_per_symbol[each_pnt]["point_features"])
                with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                    dump(features_per_symbol[each_pnt], f, ensure_ascii=False)
                    # else:
                    #     if cmp_eval:
                    #         each_pnt = pnt_pair_per_map[each_pnt]
                    #         print('saving point symbol outputs named with', each_pnt)
                    #         geodict_to_raster(feature_collection,each_pnt,map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=False)
                # else:
                #     stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
                #     if not os.path.exists(stitch_output_dir_per_map):
                #         os.makedirs(stitch_output_dir_per_map) 
                #     feature_collection = FeatureCollection({})
                #     if not save_raster:    
                #         empty_output =map_name+'_strike_'+'empty'+'.geojson' 
                #         output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,empty_output)
                #         with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                #                     dump(feature_collection, f, ensure_ascii=False)
                #     # else:
                #     #     geodict_to_raster(feature_collection,'empty',map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=True)
        else:
            stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
            if not os.path.exists(stitch_output_dir_per_map):
                os.makedirs(stitch_output_dir_per_map) 
            feature_collection = FeatureCollection({})
            if not save_raster:    
                empty_output =map_name+'_'+'_strike_'+'.geojson' 
                output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,empty_output)
                with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                            dump(feature_collection, f, ensure_ascii=False)
            # else:
            #     geodict_to_raster(feature_collection,'empty',map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=True)



    
    else:
        stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
        if not os.path.exists(stitch_output_dir_per_map):
            os.makedirs(stitch_output_dir_per_map) 
        feature_collection = FeatureCollection({})
        if not save_raster:    
            empty_output =map_name+'_strike_'+'empty'+'.geojson' 
            output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,empty_output)
            with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                        dump(feature_collection, f, ensure_ascii=False)
        # else:
        #     geodict_to_raster(feature_collection,'empty',map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=True)





