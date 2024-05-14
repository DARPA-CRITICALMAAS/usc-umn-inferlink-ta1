import pandas as pd
import numpy as np
from PIL import Image
import cv2
from utils import get_toponym_tokens, prepare_bm25, fuzzy_find_top_k_matching
from segmentation_sam import resize_img, run_sam
import urllib.request
import rasterio
import json 
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

# torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

api_key = os.getenv("OPENAI_API_KEY")


def run_segmentation(file, support_data_dir):
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
        "model": "gpt-4-vision-preview",
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


def run_georeferencing(args):

    # input_path = '../input_data/CO_Frisco.png' # supported file format: png, jpg, jpeg, tif
    input_path = args.input_path
    support_data_dir = args.support_data_dir
    temp_dir  = args.temp_dir

    # load data store in bm25 & df_merged
    bm25, df_merged = load_data(topo_histo_meta_path = f'{support_data_dir}/historicaltopo.csv',
        topo_current_meta_path = f'{support_data_dir}/ustopo_current.csv')
    
    seg_mask, seg_bbox, image, image_width, image_height = run_segmentation(input_path, args.support_data_dir)

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


# def download_geotiff(geotiff_url, temp_dir):
#     filename = os.path.basename(geotiff_url)
#     try:
#         print(f'Downloading {filename}')
#         urllib.request.urlretrieve(geotiff_url, os.path.join(temp_dir,filename))
#         return os.path.join(temp_dir,filename)
#     except URLError as e:
#         print(f'Error during download: {e}')
#         return -1

# def img2geo_georef(row_col_list, geotiff_file):
#     # Open the GeoTIFF file
#     topo_target_gcps = []
#     with rasterio.open(geotiff_file) as dataset:
#         # Use the dataset's transform attribute and the `xy` method
#         # to convert pixel coordinates to geographic coordinates.
#         # The `xy` method returns the coordinates of the center of the given pixel.
#         # geotransform = dataset.transform

#         for row, col in row_col_list:
#             geo_coords = dataset.xy(col, row, offset='center')
#             topo_target_gcps.append(geo_coords)


#         # transform = dataset.transform
#         # for row, col in row_col_list:
#         #     lon, lat = transform * (col, row)
#         #     topo_target_gcps.append((lat, lon))

#     # print(dataset.crs.wkt)

#     return topo_target_gcps, dataset.crs.wkt 
    
# def get_gcps(args, geologic_seg_bbox, top10):
#     top1 = top10.iloc[0]

#     # left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
#     left, right, top, bottom = geologic_seg_bbox[0], geologic_seg_bbox[0]+geologic_seg_bbox[2], geologic_seg_bbox[1], geologic_seg_bbox[1]+geologic_seg_bbox[3]

#     geologic_src_gcps = [(top, left),(top, right),(bottom, right), (bottom, left)]

#     geotiff_url = top1['geotiff_url']

#     geotiff_path = download_geotiff(geotiff_url, args.temp_dir) 

#     if geotiff_path == -1: #failure if -1
#         return -1 

#     seg_mask, topo_seg_bbox, image, image_width, image_height = run_segmentation(geotiff_path, args.support_data_dir)
#     topo_left, topo_right, topo_top, topo_bottom = topo_seg_bbox[0], topo_seg_bbox[0]+topo_seg_bbox[2], topo_seg_bbox[1], topo_seg_bbox[1]+topo_seg_bbox[3]

#     topo_src_gcps = [(topo_top, topo_left),(topo_top, topo_right),(topo_bottom, topo_right), (topo_bottom, topo_left)]
#     topo_target_gcps, dataset_crs = img2geo_georef(topo_src_gcps, geotiff_path)
    
#     return geologic_src_gcps, topo_target_gcps, dataset_crs


# def generate_geotiff(args, geologic_seg_bbox, top10, width, height,):

#     input_tiff = args.input_path
#     temp_tiff = os.path.join(args.temp_dir,os.path.basename(args.input_path))

#     topo_src_gcps, topo_target_gcps, dataset_crs = get_gcps(args, geologic_seg_bbox, top10) 

#     command = 'gdal_translate -of GTiff' 
#     for src_gcp, target_gcp in zip(topo_src_gcps, topo_target_gcps):
#         command +=  ' -gcp ' + str(src_gcp[0]) + ' ' + str(src_gcp[1]) + ' '
#         command += str(target_gcp[0]) + ' ' + str(target_gcp[1]) + ' '

#     command = command + input_tiff + ' ' + temp_tiff 

#     print(command)
#     if os.path.exists(temp_tiff):
#         os.remove(temp_tiff)
#     os.system(command)
    
#     output_tiff = os.path.join(args.temp_dir, os.path.basename(args.input_path).split('.')[0] + '_geo' + '.tif') 
#     command1 = "gdalwarp -r near -t_srs '" + dataset_crs + "' -of GTiff " + temp_tiff + " " + output_tiff # extra single quote before and after dataset crs

#     print(command1)
#     if os.path.exists(output_tiff):
#         os.remove(output_tiff)
#     os.system(command1)
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



def write_to_json(args, seg_bbox, top10, width, height, title, toponyms):

    top1 = top10.iloc[0]
    # print(top1['product_url'])
    
    left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
    # img_left, img_right, img_top, img_bottom = seg_bbox[0], seg_bbox[0]+seg_bbox[2], seg_bbox[1], seg_bbox[1]+seg_bbox[3] # for SAM axis aligned bbox
    reformatted_bbox = reformat_bbox(seg_bbox, row_first = False)
    px_points = [reformatted_bbox['top_left'],reformatted_bbox['top_right'], reformatted_bbox['bottom_right'],reformatted_bbox['bottom_left']]

    geo_points = [[left,top],[right, top],[right, bottom],[left,bottom]]

    print(geo_points)
    print(px_points)

    if args.reformat:
        # top10_dict = top10.to_dict(orient='records') 

        # Use increasing numbers as keys
        # top10_dict = {i + 1: row for i, row in enumerate(top10_dict)}
        # georef_output_dict =   { 
        #              "width": width, 
        #              "height": height, 
        #              "title": title, 
        #              "toponyms": toponyms, 
        #              "seg_bbox": seg_bbox, 
        #             "geo":
        #                 {"left":left, "right":right, "top":top, "bottom":bottom},
        #             "img":
        #                 {"img_left":img_left, "img_right":img_right, "img_top":img_top, "img_bottom":img_bottom},
        #             "top10": top10_dict,
        #         }

        gcp_list = []
        
        for i in range(4):
            cur_gcp_dict = {"id": i+1} 
            cur_gcp_dict["map_geom"] = geo_points[i]
            cur_gcp_dict["px_geom"] = px_points[i]
            cur_gcp_dict["confidence"] = None
            cur_gcp_dict["provenance"] = "modelled"
            gcp_list.append(cur_gcp_dict)


        georef_output_dict = {
            "map":{
                "name": os.path.basename(args.input_path).rsplit('.', 1)[0],
                "projection_info": {"projection": "EPSG:4326", "provenance":"modelled",
                "gcps": gcp_list}
            }
        }

        with open(args.output_path, 'w') as f:
            json.dump([georef_output_dict], f)

    else:
        gcp_list = []
        
        for i in range(4):
            cur_gcp_dict = {"gcp_id": i+1} 
            cur_gcp_dict["map_geom"] = {"latitude":geo_points[i][1], "longitude":geo_points[i][0]}
            cur_gcp_dict["px_geom"] = {"columns_from_left":px_points[i][1], "rows_from_top":px_points[i][0]} 
            cur_gcp_dict["confidence"] = None
            cur_gcp_dict["model"] = "umn"
            cur_gcp_dict["model_version"] = "0.0.1"
            cur_gcp_dict["crs"] = "EPSG:4326"
            gcp_list.append(cur_gcp_dict)

        
        georef_output_dict = {
            "cog_id": None,
            # "map_name": os.path.basename(args.input_path).rsplit('.', 1)[0],
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




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None) 
    parser.add_argument('--temp_dir', type=str, default='temp') 
    parser.add_argument('--support_data_dir', type=str, default='support_data')
    parser.add_argument('--reformat', action='store_true' )
    parser.add_argument('--more_info', action='store_true' )
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    assert args.input_path is not None
    assert args.output_path is not None 
    
    seg_bbox, top10, image_width, image_height, title, toponyms = run_georeferencing(args)

    ret = run_georeferencing(args)

    if ret == -1:
        return -1 
    else:
        seg_bbox, top10, image_width, image_height, title, toponyms = ret

    # generate_geotiff(args, seg_bbox, top10, image_width, image_height)

    # write_to_geopackage(args, seg_bbox, top1, image_width, image_height)
    write_to_json(args, seg_bbox, top10, image_width, image_height, title, toponyms)

    return 0 

    
if __name__ == '__main__':

    main()

'''
Example command:
to json:
python3 run_georeference.py --input_path='/home/zekun/data/nickel_0209/raw_maps/169_34067.tif' --output_path='temp/debug.json' --support_data_dir='/home/zekun/ta1_georeferencing/geological-map-georeferencing/support_data'

python3 run_georeference_moreinfo.py --input_path='/home/zekun/data/nickel_0209/raw_maps/169_34067.tif' --output_path='/home/zekun/data/nickel_output/0209/169_34067.json'

to geopackage:
python3 run_georeference_moreinfo.py  --input_path='../input_data/CO_Frisco.png' --output_path='../output_georef/CO_Frisco.gpkg'
'''
