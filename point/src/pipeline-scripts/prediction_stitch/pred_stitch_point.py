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


def predict_img_patches(map_name,crop_dir_path,model_dir_root,selected_models,predict_output_dir):

    # map_name = os.path.basename(os.path.dirname(crop_dir_path))
    output_path=os.path.join(predict_output_dir,map_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)     
    print(output_path)
    for val_img in os.listdir(crop_dir_path):            
        # if val_img.endswith('.jpg') or val_img.endswith('.png'):
        entire_res=[] 
        img_path=os.path.join(crop_dir_path,val_img)  
        for idx,model_file in enumerate(selected_models):
            weight_path=os.path.join(model_dir_root,model_file)
            print(weight_path)
            # print(torch.load(weight_path))
            model = YOLO(weight_path)
            pnt_name=model_file.split('.')[0]                 
            results = model(img_path,conf=0.25)  # results list
            res_boxes = results[0].boxes.data.cpu().numpy()             
            for i, box in enumerate(res_boxes):
                res_per_crop={}
                res_per_crop['img_geometry']=[]
                res_per_crop['type']=None
                res_per_crop['score']=[]
                res_per_crop['bbox']=[]
                x1, y1, x2, y2, conf, _ = box
                cnt_x=int((x1+x2)/2)
                cnt_y=int((y1+y2)/2)
                res_per_crop['img_geometry'].append([cnt_x,cnt_y])
                res_per_crop['type']=pnt_name
                res_per_crop['score']=str(conf)
                res_per_crop['bbox'].append([int(x1), int(y1), int(x2), int(y2)])

                entire_res.append(res_per_crop)                   
            
            out_file_path=os.path.join(output_path,val_img.split('.')[0]+'.json')
            with open(out_file_path, "w") as out:
                json.dump(entire_res, out)

def write_to_tif(out_file_path, output_tif):
    output_tif = output_tif.astype(np.uint8)
    im = Image.fromarray(output_tif) 
    im.save(out_file_path, "TIFF")
    return 


def geodict_to_raster(geodict,pnt_name,map_name,map_sheet_dir,raster_output_dir, empty_flag):
    print('++===LLLL', os.path.join(raster_output_dir,map_name +'_'+pnt_name+'.tif'))
    #input: geodict : dictionary per each symbol
    map_sheet_file = map_name + ".tif"
    file_path = os.path.join(map_sheet_dir, map_name, map_sheet_file)
    tif_image = Image.open(file_path)
    width, height = tif_image.size

    raster_layer_per_pnt = np.zeros((height,width))
    if not empty_flag:
        point_features_list = geodict['features']     
        for point_feature_info in point_features_list:
            coord = point_feature_info['geometry']['coordinates']
            x, y = int(coord[0]), int(coord[1])
            raster_layer_per_pnt[y,x] = 1
    write_to_tif(os.path.join(raster_output_dir,map_name +'_'+pnt_name+'.tif'), raster_layer_per_pnt)

def stitch_to_each_point(map_name, crop_dir_path,pred_root,stitch_root, map_sheets_dir ,save_raster, cmp_eval, pnt_pair_per_map ): 
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
                new_bbox = [line_data['bbox'][0][0]+patch_index_w, line_data['bbox'][0][1]+ patch_index_h, line_data['bbox'][0][2] + patch_index_w, line_data['bbox'][0][3]+patch_index_h]
                line_data['bbox'][0] = new_bbox
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
            sym_type = sym_type.split('.')[0]
            score = line_data['score']
            if len(line_data['bbox'][0]) == 4:
                x1,y1,x2,y2 = line_data['bbox'][0]
                bbox=[x1,y1,x2,y2]
            else:
                bbox=[0,0,0,0]
            if sym_type not in features_per_symbol.keys():
                features_per_symbol[sym_type] = []
            features_per_symbol[sym_type].append(Feature(geometry = point, properties={'type': sym_type, "id": len(features_per_symbol[sym_type]), "score": score, "bbox": bbox ,"dip" : 0 ,"dip_direction" : 0.0, "provenance": "modelled" }))

        stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
        if not os.path.exists(stitch_output_dir_per_map):
            os.makedirs(stitch_output_dir_per_map) 
        if len(features_per_symbol) != 0:
            for each_pnt in features_per_symbol.keys():
                if len(features_per_symbol[each_pnt]) != 0:
                    feature_collection = FeatureCollection(features_per_symbol[each_pnt])
                    if not save_raster:
                        each_file_per_pnt_name=map_name+'_'+each_pnt+'.geojson'
                        output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,each_file_per_pnt_name)              
                        with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                            dump(feature_collection, f, ensure_ascii=False)
                    else:
                        if cmp_eval:
                            each_pnt = pnt_pair_per_map[each_pnt]
                            print('saving point symbol outputs named with', each_pnt)
                            geodict_to_raster(feature_collection,each_pnt,map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=False)
                else:
                    stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
                    if not os.path.exists(stitch_output_dir_per_map):
                        os.makedirs(stitch_output_dir_per_map) 
                    feature_collection = FeatureCollection({})
                    if not save_raster:    
                        empty_output =map_name+'_'+'empty'+'.geojson' 
                        output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,empty_output)
                        with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                                    dump(feature_collection, f, ensure_ascii=False)
                    else:
                        geodict_to_raster(feature_collection,'empty',map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=True)
        else:
            stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
            if not os.path.exists(stitch_output_dir_per_map):
                os.makedirs(stitch_output_dir_per_map) 
            feature_collection = FeatureCollection({})
            if not save_raster:    
                empty_output =map_name+'_'+'empty'+'.geojson' 
                output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,empty_output)
                with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                            dump(feature_collection, f, ensure_ascii=False)
            else:
                geodict_to_raster(feature_collection,'empty',map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=True)



    
    else:
        stitch_output_dir_per_map = os.path.join(stitch_root, map_name)
        if not os.path.exists(stitch_output_dir_per_map):
            os.makedirs(stitch_output_dir_per_map) 
        feature_collection = FeatureCollection({})
        if not save_raster:    
            empty_output =map_name+'_'+'empty'+'.geojson' 
            output_geojson_per_pnt = os.path.join(stitch_output_dir_per_map,empty_output)
            with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                        dump(feature_collection, f, ensure_ascii=False)
        else:
            geodict_to_raster(feature_collection,'empty',map_name,map_sheets_dir,stitch_output_dir_per_map, empty_flag=True)





