import pandas as pd
import numpy as np
from PIL import Image
import cv2
from utils import get_toponym_tokens, prepare_bm25, fuzzy_find_top_k_matching
from segmentation_sam import resize_img, run_sam
import urllib.request
import rasterio
import json 
import pdb
import numpy as np
import torch
import base64
import requests
import argparse
import logging 
import time 
import csv
import io
import os


Image.MAX_IMAGE_PIXELS = None

api_key = os.getenv("OPENAI_API_KEY")


def run_segmentation(file, support_data_dir, device):
    image = Image.open(file).convert('RGB') # read image with PIL library

    print('image_size',image.size)
    resized_img, scaling_factor = resize_img(np.array(image))
    seg_mask, bbox, rotated_bbox_points = run_sam(np.array(image), resized_img, scaling_factor, device, support_data_dir)

    return seg_mask, rotated_bbox_points, image, image.width, image.height


def load_data(topo_histo_meta_path, topo_current_meta_path):

    # topo_histo_meta_path = 'support_data/historicaltopo.csv'
    df_histo = pd.read_csv(topo_histo_meta_path) 
            
    # topo_current_meta_path = 'support_data/ustopo_current.csv'
    df_current = pd.read_csv(topo_current_meta_path) 

    # common_columns = df_current.columns.intersection(df_histo.columns)

    # df_merged = pd.concat([df_histo[common_columns], df_current[common_columns]], axis=0)
    df_merged = df_histo #TODO: tempory fix to get Geotiff URL

    bm25 = prepare_bm25(df_merged)

    return bm25, df_merged


def get_topo_basemap(query_sentence, bm25, df_merged, device ):

    print(query_sentence)
    query_tokens, human_readable_tokens = get_toponym_tokens(query_sentence, device)
    print(human_readable_tokens)

    query_sent = ' '.join(human_readable_tokens)

    topk = fuzzy_find_top_k_matching(query_sent, df_merged, k=10)
    fuzzy_top10 = df_merged.iloc[[a[0] for a in topk]]

    tokenized_query = query_sent.split(" ")
    
    doc_scores = bm25.get_scores(tokenized_query)

    sorted_bm25_list = np.argsort(doc_scores)[::-1]
    
    # Top 10
    top10 = df_merged.iloc[sorted_bm25_list[0:10]]

    # top1 = df_merged.iloc[sorted_bm25_list[0]]

    # return top10, query_sent 
    return fuzzy_top10, query_sent


# Function to encode the image
def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_image

def getTitle(base64_image, max_trial = 10):

    if base64_image is None:
        return "No file selected"


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4o",
        "messages": [
          {
            "role": "user",
            "content": [
              {
                "type": "text",
                "text": "What’s the title of map? please just return the title no more words else"
              },
              {
                "type": "image_url",
                "image_url": {
                  "url": f"data:image/jpeg;base64,{base64_image}"
                }
              }
            ]
          }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
    cnt_trial = 1

    try: 
        while ('choices' not in response and cnt_trial < max_trial):
            time.sleep(5) # sleep for 5 seconds before sending the next request
            print('Title extraction failed, retrying...')
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json() 
            cnt_trial += 1 
    except Exception as e:
        print(e)

    if 'choices' in response:
        return response['choices'][0]['message']['content']
    else:
        return -1



# check and downscale:

def downscale(image, max_size=10, max_dimension=9500):

    print("downscaling...")

    buffer = io.BytesIO()

    image.save(buffer, format="JPEG")  

    img_size = buffer.tell() / (1024 * 1024)

    if img_size > max_size or max(image.width, image.height) > max_dimension:

        downscale_factor = max_size / img_size

        downscale_factor = max(downscale_factor, 0.1)

        new_size = (int(image.width * downscale_factor), int(image.height * downscale_factor))

        while True:

            # to aviod the case err "Maximum supported image dimension is 65500 pixels"
            while max(image.width, image.height) > max_dimension:
                print("---")
                downscale_factor = max_dimension / max(image.width, image.height)
                downscale_factor = max(downscale_factor, 0.1)
                new_size = (int(image.width * downscale_factor), int(image.height * downscale_factor))
                image=image.resize(new_size)

            print("out")

            downscaled_img = image.resize(new_size)
            buffer = io.BytesIO()
            downscaled_img.save(buffer, format="JPEG")
            downscaled_size = buffer.tell() / (1024 * 1024)

            print(downscaled_size)

            print("downscaled x 1")

            if downscaled_size < max_size or max(downscaled_img.width, downscaled_img.height) < max_dimension:
                print("dimension now:")
                print(max(downscaled_img.width, downscaled_img.height))
                print("down.")
                break  
            else:
                downscale_factor *= 0.8 


            new_size = (int(image.width * downscale_factor), int(image.height * downscale_factor))

        print("after downscaled, the new size is: ")
        print(new_size)

        downscaled_img = image.resize(new_size)

        return downscaled_img

    else:

        return image


def to_camel(title):
    words = title.split()
    return ' '.join(word.capitalize() for word in words)


def run_georeferencing_gpt(args):

    # Check if GPU is available
    if torch.cuda.is_available():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        # Assign GPU id (assuming you want to use the first GPU)
        gpu_id = int(args.gpu_id)
        print(f"Using GPU {gpu_id}")

        # Set the device
        device = torch.device(f'cuda:{gpu_id}')
    else:
        print("GPU is not available. Using CPU instead.")
        # If GPU is not available, use CPU
        device = torch.device('cpu')

    # input_path = '../input_data/CO_Frisco.png' # supported file format: png, jpg, jpeg, tif
    input_path = args.input_path
    support_data_dir = args.support_data_dir
    temp_dir  = args.temp_dir

    # load data store in bm25 & df_merged
    bm25, df_merged = load_data(topo_histo_meta_path = f'{support_data_dir}/historicaltopo.csv',
        topo_current_meta_path = f'{support_data_dir}/ustopo_current.csv')
    
    seg_mask, seg_bbox, image, image_width, image_height = run_segmentation(input_path, args.support_data_dir, device)

    # Set the opacity level (0.5 for 50% opacity)
    opacity = 0.5

    seg_mask_channel3 = cv2.merge((seg_mask,seg_mask,seg_mask))

    # Perform the image overlay
    overlay = cv2.addWeighted(np.asarray(image), opacity, seg_mask_channel3, 1 - opacity, 0)
    
    cv2.imwrite(os.path.join(args.temp_dir, os.path.basename(input_path).split('.')[0]+'.jpg'), overlay)

    if not os.path.isdir(args.temp_dir):
        os.makedirs(args.temp_dir)
    
    jpg_file_path = os.path.join(args.temp_dir, "output.jpg")

    image.save(jpg_file_path, format="JPEG")

    image =downscale(image)
    # Getting the base64 string
    base64_image = encode_image(image)
    title = getTitle(base64_image)

    if title == -1: # exception when extracting the title 
        logging.error('Failed to extract title, exit with code -1')
        return -1 

    title = to_camel(title)

    os.remove(jpg_file_path)

    query_sentence = title

    # get the highest score: 
    top10, toponyms = get_topo_basemap(query_sentence, bm25, df_merged, device )

    return seg_bbox, top10, image_width, image_height, title, toponyms 


"""
min_bbox_point: list of points, containing the rotated bounding box for map segmentation area, in the format of [row,col]
row_first: output reformated bbox in [row,col] format, otherwise [col,row]

"""
def reformat_bbox(min_bbox_points, row_first = True):

    # Find the indices of the points with the minimum and maximum sum of coordinates
    top_left_index = np.argmin(np.sum(min_bbox_points, axis=1))
    bottom_right_index = np.argmax(np.sum(min_bbox_points, axis=1))

    # Find the indices of the points with the minimum and maximum difference of coordinates
    top_right_index = np.argmax(np.diff(min_bbox_points, axis=1))
    bottom_left_index = np.argmin(np.diff(min_bbox_points, axis=1))

    # Assign the identified points
    top_left = min_bbox_points[top_left_index]
    top_right = min_bbox_points[top_right_index]
    bottom_right = min_bbox_points[bottom_right_index]
    bottom_left = min_bbox_points[bottom_left_index]

    if row_first:
        ret_bbox =  {'top_left':top_left, 'top_right':top_right, 'bottom_right':bottom_right, 'bottom_left':bottom_left}
    else:
        ret_bbox = {'top_left':[top_left[1], top_left[0]], 'top_right':[top_right[1], top_right[0]], 
                    'bottom_right':[bottom_right[1], bottom_right[0]], 'bottom_left':[bottom_left[1], bottom_left[0]]}

    return ret_bbox 

def construct_gcp_dict(geo_points, px_points):
    gcp_list = []
    for i in range(4):
        cur_gcp_dict = {"gcp_id": i+1} 
        cur_gcp_dict["map_geom"] = {"latitude":geo_points[i][0], "longitude":geo_points[i][1]}
        cur_gcp_dict["px_geom"] = {"columns_from_left":px_points[i][0], "rows_from_top":px_points[i][1]} 
        cur_gcp_dict["confidence"] = None
        cur_gcp_dict["model"] = "umn"
        cur_gcp_dict["model_version"] = "0.0.1"
        cur_gcp_dict["crs"] = "EPSG:4326"
        gcp_list.append(cur_gcp_dict)

    return gcp_list 

def write_to_json_gpt(args, seg_bbox, top10, width, height, title, toponyms):

    top1 = top10.iloc[0]
    # print(top1['product_url'])
    
    left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
    # img_left, img_right, img_top, img_bottom = seg_bbox[0], seg_bbox[0]+seg_bbox[2], seg_bbox[1], seg_bbox[1]+seg_bbox[3] # for SAM axis aligned bbox
    reformatted_bbox = reformat_bbox(seg_bbox, row_first = False)
    px_points = [reformatted_bbox['top_left'],reformatted_bbox['top_right'], reformatted_bbox['bottom_right'],reformatted_bbox['bottom_left']]

    # px_points = [[a[1], a[0]] for a in px_points]

    # geo_points = [[left,top],[right, top],[right, bottom],[left,bottom]]
    geo_points = [[top,left],[top, right],[bottom, right],[bottom,left]]

    gcp_list = construct_gcp_dict(geo_points, px_points)

    if args.more_info:
        
        georef_output_dict = {
            "cog_id": None,
            "gpt_title": title,
            "toponyms": toponyms, 
            "georeference_results":[
                {
                    "likely_CRSs":["EPSG:4326"],
                    "map_area": None,
                    "projections": None,
                },
            ],
            "gcps": gcp_list,
            "system":"umn",
            "system_version":"0.0.1"
        }

        with open(args.output_path, 'w') as f:
            json.dump(georef_output_dict, f, indent = 4)

    # elif args.reformat:

    #     gcp_list = []
        
    #     for i in range(4):
    #         cur_gcp_dict = {"id": i+1} 
    #         cur_gcp_dict["map_geom"] = geo_points[i]
    #         cur_gcp_dict["px_geom"] = px_points[i]
    #         cur_gcp_dict["confidence"] = None
    #         cur_gcp_dict["provenance"] = "modelled"
    #         gcp_list.append(cur_gcp_dict)


    #     georef_output_dict = {
    #         "map":{
    #             "name": os.path.basename(args.input_path).rsplit('.', 1)[0],
    #             "projection_info": {"projection": "EPSG:4326", "provenance":"modelled",
    #             "gcps": gcp_list}
    #         }
    #     }

    #     with open(args.output_path, 'w') as f:
    #         json.dump([georef_output_dict], f)

    else:
        
        georef_output_dict = {
            "cog_id": None,
            "georeference_results":[
                {
                    "likely_CRSs":["EPSG:4326"],
                    "map_area": None,
                    "projections": None,
                },
            ],
            "gcps": gcp_list,
            "system":"umn",
            "system_version":"0.0.1"
        }

        with open(args.output_path, 'w') as f:
            json.dump(georef_output_dict, f, indent = 4)

def write_to_json_geocoords(map_id, gcp_list):

    
    georef_output_dict = {
        "cog_id": map_id,
        "georeference_results":[
            {
                "likely_CRSs":["EPSG:4326"],
                "map_area": None,
                "projections": None,
            },
        ],
        "gcps": gcp_list,
        "system":"umn",
        "system_version":"0.0.1"
    }

    return georef_output_dict

def read_map_content_area_from_json(legend_json_path):#
    map_area_bbox = None
    
    try:   
        with open(legend_json_path, 'r', encoding='utf-8') as file:
            legend_dict = json.load(file)
    except:
        return FileNotFoundError(f'{legend_json_path} does not exist')

    for item in legend_dict['segments']:
        if 'map' == item['class_label']:
            map_area_bbox = item['bbox'] 
        
    return map_area_bbox

def dms_to_decimal(dms):
    # Handle cases with degrees, minutes, and seconds
    parts = dms.split('°')
    degrees = int(parts[0])
    
    if "'" in parts[1]:
        minutes_part = parts[1].split("'")
        minutes = float(minutes_part[0])
        seconds = 0.0
        if '"' in minutes_part[1]:
            seconds = float(minutes_part[1].strip('"'))
    else:
        minutes = 0.0
        seconds = 0.0
    
    decimal = degrees + minutes / 60 + seconds / 3600
    return decimal


def run_georeferencing_mapkurator_coords(args):

    if args.segmentation_json_path is  None :
        return -1
    if  args.mapkurator_coords_dir is None :
        return -1 

    segmentation_json_path = args.segmentation_json_path 
    mapkurator_coords_dir = args.mapkurator_coords_dir
    
    map_area_bbox = read_map_content_area_from_json(segmentation_json_path)

    if map_area_bbox is None:
        return -2 # segmentation file does not exist 

    # x: height, y: width
    map_content_y, map_content_x,  map_content_w, map_content_h = map_area_bbox

    bounding_box = [(map_content_y, map_content_x ),
        (map_content_y, map_content_x + map_content_h ),
        (map_content_y + map_content_w, map_content_x + map_content_h ),
        (map_content_y + map_content_w, map_content_x )] 


    corner_texts_dict = dict()
    json_file_paths = sorted(os.listdir(mapkurator_coords_dir))
    map_id = os.path.basename(mapkurator_coords_dir)
    for json_path in json_file_paths:
        
        crop_id = json_path.split('_')[1].split('.json')[0]
        with open(os.path.join(mapkurator_coords_dir, json_path), 'r') as f:
            mapkurator_json = json.load(f)
        
        text = mapkurator_json['text'] 
        if len(text) == 0: continue 

        corner_texts_dict[crop_id] = text 


    success_flag = False 
    # heuristics with a set of rules 

    if len(corner_texts_dict) !=  4:
        return -3 # missing corner values 

    
    lat_long_sets = set()
    two_num_flag = True 
    for crop_id, values in corner_texts_dict.items():
        if len(values) != 2: 
            two_num_flag = False # TODO: can be relaxed
            continue 
        for dummy_k, geocoord in values.items():
            lat_long_sets.add(geocoord)

    

    if len(lat_long_sets) == 4 and two_num_flag == True: # heuristics, only 2 unique values for lat and 2 unique values for long 
        print(corner_texts_dict)
        success_flag = True 

    if success_flag == True: 
        
        geo_points = []
        px_points = []
        for i in range(4):
            texts = corner_texts_dict[str(i)]

            point = bounding_box[i]
            x, y = map(int, point) # TA4 Jataware's x y is fliped from ours 

            decimals = [dms_to_decimal(t) for k, t in texts.items()]
            lat, lon = min(decimals),max(decimals)

            geo_points.append([lat, -lon])
            px_points.append([x, y])

        gcp_list = construct_gcp_dict(geo_points, px_points)

        json_data = write_to_json_geocoords(map_id, gcp_list)

        with open(args.output_path, 'w') as f:
            json.dump(json_data, f, indent=4)
            
        return 0 
    else:
        return -3 # did not pass the heuristics


# todo: convert to mitre format 



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None) 
    parser.add_argument('--temp_dir', type=str, default='temp') 
    parser.add_argument('--support_data_dir', type=str, default='support_data')

    # args for mapkurator-coords
    parser.add_argument('--mapkurator_coords_dir', type=str, default=None)
    parser.add_argument('--segmentation_json_path', type = str, default = None)


    # args for converting to competition output format 
    parser.add_argument('--competition_form', action='store_true' )
    parser.add_argument('--clue_file', type = str, default = None)

    # parser.add_argument('--reformat', action='store_true' )
    parser.add_argument('--more_info', action='store_true' )
    parser.add_argument('--gpu_id', type=str, default='')
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    assert args.input_path is not None
    assert args.output_path is not None 

    

    json_path = '../mapkurator_coords/spotting_output_validation_corner_crops.json' 
    uncharted_path = 'min_bbox_json/uncharted_min_bbox_validation.json'

    ret_geocoords = run_georeferencing_mapkurator_coords(args)
    print('mapkurator-coords return code:', ret_geocoords)

    if ret_geocoords != 0: # failed to handle with mapkurator-coords approach 
        logging.info('mapKurator-coords approach fails to handle map, switch to GPT-toponym')
    
        ret = run_georeferencing_gpt(args)

        if ret == -1:
            return -1 
        else:
            seg_bbox, top10, image_width, image_height, title, toponyms = ret
        
        print(ret)

        write_to_json_gpt(args, seg_bbox, top10, image_width, image_height, title, toponyms)


    return 0 

    
if __name__ == '__main__':

    main()

'''
Example command:
python3 run_georeference.py \ 
--input_path='/home/yaoyi/shared/critical-maas/9month/raw_maps/330e339d3b0d335d26bd263d361d361d270d264e161e375e274e2f5e0b5e9198.cog.tif' \
--output_path='debug/debug.geojson' \ 
--temp_dir='temp' \ 
--support_data_dir='/home/yaoyi/li002666/critical_maas/support_data/' \ 
--mapkurator_coords_dir='coordinate_spotting/<COG_ID>/spotter/<COG_ID>' \ 
--segmentation_json_path='legend_segment/<COG_ID>_map_segmentation.json' 
'''
