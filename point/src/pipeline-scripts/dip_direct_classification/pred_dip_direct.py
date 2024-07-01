#strike classification script
'''
strike angle extraction model 
'''
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import argparse
import json
import shutil
from pathlib import Path
Image.MAX_IMAGE_PIXELS = 933120000

def predict_strike_per_inst(model,each_crop_img,device):
    dip_direction =  0
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    model.eval()
    # image = Image.open(image_path).convert("RGB")
    each_crop_img = transform(each_crop_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(each_crop_img).cpu().numpy().flatten()
    
    angle_rad = np.arctan2(output[0], output[1])
    dip_direction = np.rad2deg(angle_rad)
    if dip_direction < 0:
        dip_direction += 360
    return dip_direction

def init_model_info(pt_model_path):
    # initialize model for strike classification
    model = models.resnet50(weights=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 360)  
    pt_ckpt = torch.load(pt_model_path,map_location='cuda:0')
    model.load_state_dict(pt_ckpt['model_state_dict'])
    model = model.to(device)
    return model,device

def pred_strike_per_pnt(map_name, map_dir, stitch_dir_per_map, final_output_dir_per_map,  dip_direct_model_path):
    
    pt_with_st = ['inclined_bedding.geojson','inclined_metamorphic.geojson','inclined_flow_banding.geojson','overturned_bedding.geojson'] 
    map_sheet_path = os.path.join(map_dir,map_name+'.tif')
    model,device =  init_model_info(dip_direct_model_path)
    stitch_list_per_map =  os.listdir(stitch_dir_per_map)
    for each_pt_geojson in stitch_list_per_map:
        each_geojson_path = os.path.join(stitch_dir_per_map,each_pt_geojson)
        if each_pt_geojson.endswith(tuple(pt_with_st)):
            print(each_geojson_path)   
            print('map_sheet_path',map_sheet_path)
            map_sheet_img = Image.open(Path(map_sheet_path),mode='r')
            f = open(each_geojson_path, 'r')    
            each_dict = json.load(f)      
            f.close() 
            new_geojson_data = each_dict.copy()
            if len(new_geojson_data["point_features"]) != 0:   
                for each_inst in new_geojson_data["point_features"]:
                    # print(each_inst['features'][0]['properties']['dip_direction'])                        
                    [x, y] = each_inst['features'][0]['geometry']['coordinates']
                    each_inst_img = map_sheet_img.crop((x-30,y-30,x+30,y+30))
                    dip_direction = predict_strike_per_inst(model,each_inst_img,device)
                    each_inst['features'][0]['properties']['dip_direction'] = str((int)(dip_direction))

                # saving output with dip direction
                output_geojson_per_pnt = os.path.join(final_output_dir_per_map,each_pt_geojson)
                with open(output_geojson_per_pnt, 'w', encoding='utf8') as f:
                    json.dump(new_geojson_data, f, ensure_ascii=False)
            else:
                shutil.copyfile(each_geojson_path, os.path.join(final_output_dir_per_map,each_pt_geojson))
        else:
            print('copying previous one')
            shutil.copyfile(each_geojson_path, os.path.join(final_output_dir_per_map,each_pt_geojson))
    

   
