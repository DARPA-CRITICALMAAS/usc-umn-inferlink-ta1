import pandas as pd
import numpy as np
from PIL import Image
import cv2
from utils import get_toponym_tokens, prepare_bm25
from segmentation_sam import resize_img, run_sam
import urllib.request
import rasterio
import json 
import numpy as np
import torch
import base64
import requests
import argparse
import csv
import io
import os


Image.MAX_IMAGE_PIXELS = None

# torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

api_key = os.getenv("OPENAI_API_KEY")


def run_segmentation(file, support_data_dir):
    image = Image.open(file) # read image with PIL library

    print('image_size',image.size)

    resized_img, scaling_factor = resize_img(np.array(image))
    seg_mask, bbox = run_sam(np.array(image), resized_img, scaling_factor, device, support_data_dir)

    return seg_mask, bbox, image, image.width, image.height


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

    tokenized_query = query_sent.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    sorted_bm25_list = np.argsort(doc_scores)[::-1]
    
    # Top 10
    top10 = df_merged.iloc[sorted_bm25_list[0:10]]

    # top1 = df_merged.iloc[sorted_bm25_list[0]]

    return top10, query_sent 


# Function to encode the image
def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_image

def getTitle(base64_image):

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

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    return response.json()['choices'][0]['message']['content']



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
    title = to_camel(getTitle(base64_image))

    os.remove(jpg_file_path)

    query_sentence = title

    # get the highest score: 
    top10, toponyms = get_topo_basemap(query_sentence, bm25, df_merged, device )

    return seg_bbox, top10, image_width, image_height, title, toponyms 

def write_to_geopackage1(seg_bbox, top1, output_path):
    db = GeopackageDatabase(
    "my_map.gpkg",
    crs="EPSG:4326" # Geographic coordinates (default)
    # crs="CRITICALMAAS:pixel" # Pixel coordinates
    )

    # Insert types (required for foreign key constraints)
    db.write_models([
    db.model.map(id="test", name="test", description="test"),
    db.model.polygon_type(id="test", name="test", description="test"),
    ])

    # Write features
    feat = {
        "properties": {
            "id": "test",
            "map_id": "test",
            "type": "test",
            "confidence": None,
            "provenance": None,
        },
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]]],
        },
    }
    db.write_features("polygon_feature", [feat])

def download_geotiff(geotiff_url, temp_dir):
    filename = os.path.basename(geotiff_url)
    try:
        print(f'Downloading {filename}')
        urllib.request.urlretrieve(geotiff_url, os.path.join(temp_dir,filename))
        return os.path.join(temp_dir,filename)
    except URLError as e:
        print(f'Error during download: {e}')
        return -1

def img2geo_georef(row_col_list, geotiff_file):
    # Open the GeoTIFF file
    topo_target_gcps = []
    with rasterio.open(geotiff_file) as dataset:
        # Use the dataset's transform attribute and the `xy` method
        # to convert pixel coordinates to geographic coordinates.
        # The `xy` method returns the coordinates of the center of the given pixel.
        # geotransform = dataset.transform

        for row, col in row_col_list:
            geo_coords = dataset.xy(col, row, offset='center')
            topo_target_gcps.append(geo_coords)


        # transform = dataset.transform
        # for row, col in row_col_list:
        #     lon, lat = transform * (col, row)
        #     topo_target_gcps.append((lat, lon))

    # import pdb # should match to lat, lon onthe map corner
    # pdb.set_trace()
    # print(dataset.crs.wkt)

    return topo_target_gcps, dataset.crs.wkt 
    
def get_gcps(args, geologic_seg_bbox, top10):
    top1 = top10.iloc[0]

    # left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
    left, right, top, bottom = geologic_seg_bbox[0], geologic_seg_bbox[0]+geologic_seg_bbox[2], geologic_seg_bbox[1], geologic_seg_bbox[1]+geologic_seg_bbox[3]

    geologic_src_gcps = [(top, left),(top, right),(bottom, right), (bottom, left)]

    geotiff_url = top1['geotiff_url']

    geotiff_path = download_geotiff(geotiff_url, args.temp_dir) 

    if geotiff_path == -1: #failure if -1
        return -1 

    seg_mask, topo_seg_bbox, image, image_width, image_height = run_segmentation(geotiff_path, args.support_data_dir)
    topo_left, topo_right, topo_top, topo_bottom = topo_seg_bbox[0], topo_seg_bbox[0]+topo_seg_bbox[2], topo_seg_bbox[1], topo_seg_bbox[1]+topo_seg_bbox[3]

    topo_src_gcps = [(topo_top, topo_left),(topo_top, topo_right),(topo_bottom, topo_right), (topo_bottom, topo_left)]
    topo_target_gcps, dataset_crs = img2geo_georef(topo_src_gcps, geotiff_path)
    
    return geologic_src_gcps, topo_target_gcps, dataset_crs


def generate_geotiff(args, geologic_seg_bbox, top10, width, height,):

    input_tiff = args.input_path
    temp_tiff = os.path.join(args.temp_dir,os.path.basename(args.input_path))

    topo_src_gcps, topo_target_gcps, dataset_crs = get_gcps(args, geologic_seg_bbox, top10) 

    command = 'gdal_translate -of GTiff' 
    for src_gcp, target_gcp in zip(topo_src_gcps, topo_target_gcps):
        command +=  ' -gcp ' + str(src_gcp[0]) + ' ' + str(src_gcp[1]) + ' '
        command += str(target_gcp[0]) + ' ' + str(target_gcp[1]) + ' '

    command = command + input_tiff + ' ' + temp_tiff 

    print(command)
    if os.path.exists(temp_tiff):
        os.remove(temp_tiff)
    os.system(command)
    
    output_tiff = os.path.join(args.temp_dir, os.path.basename(args.input_path).split('.')[0] + '_geo' + '.tif') 
    command1 = "gdalwarp -r near -t_srs '" + dataset_crs + "' -of GTiff " + temp_tiff + " " + output_tiff # extra single quote before and after dataset crs

    print(command1)
    if os.path.exists(output_tiff):
        os.remove(output_tiff)
    os.system(command1)

    

def write_to_json(args, seg_bbox, top10, width, height, title, toponyms):

    top1 = top10.iloc[0]
    
    left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
    img_left, img_right, img_top, img_bottom = seg_bbox[0], seg_bbox[0]+seg_bbox[2], seg_bbox[1], seg_bbox[1]+seg_bbox[3]
    
    if args.more_info:
        top10_dict = top10.to_dict(orient='records') 

        # Use increasing numbers as keys
        top10_dict = {i + 1: row for i, row in enumerate(top10_dict)}
        bounds =   { 
                     "width": width, 
                     "height": height, 
                     "title": title, 
                     "toponyms": toponyms, 
                     "seg_bbox": seg_bbox, 
                    "geo":
                        {"left":left, "right":right, "top":top, "bottom":bottom},
                    "img":
                        {"img_left":img_left, "img_right":img_right, "img_top":img_top, "img_bottom":img_bottom},
                    "top10": top10_dict,
                }
    else:
        bounds =  { "geo":
                        {"left":left, "right":right, "top":top, "bottom":bottom},
                    "img":
                        {"img_left":img_left, "img_right":img_right, "img_top":img_top, "img_bottom":img_bottom},
                }

    with open(args.output_path, 'w') as f:
        json.dump(bounds, f)


def write_to_geopackage(args, seg_bbox, top1, width, height):
    input_path = args.input_path
    output_path = args.output_path

    map_name = os.path.basename(output_path).split('.')[0]
        
    db = GeopackageDatabase(
        output_path,
        crs="EPSG:4326" # Geographic coordinates (default)
        # crs="CRITICALMAAS:pixel" # Pixel coordinates
    )

    # Insert types (required for foreign key constraints)
    # TODO: source_url should be the link in NGMDB?
    db.write_models([
        db.model.map(id=map_name, name=map_name, source_url=input_path, image_url=input_path, image_width = width, image_height=height ),
    ])


    # "bounds": { "geo":
    #                 {"left":left, "right":right, "top":top, "bottom":bottom},
    #             "img":
    #                 {"left":seg_bbox[0], "right":seg_bbox[0]+seg_bbox[2], "top":seg_bbox[1], "bottom":seg_bbox[1]+seg_bbox[3]}
    #         }

    left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
    img_left, img_right, img_top, img_bottom = seg_bbox[0], seg_bbox[0]+seg_bbox[2], seg_bbox[1], seg_bbox[1]+seg_bbox[3]
    
    feat = {
         "properties": {
            "id": "0",
            "provenance": "Unknown",
            "map_id": map_name,
            "projection": "Unknown",
            # "bounds": [(img_top, img_left), (img_top, img_right), (img_bottom, img_right), (img_bottom, img_left), (img_top, img_left)]
        },
        "geometry":{
            "type": "Polygon",
            "coordinates": [[(top, left), (top, right), (bottom, right), (bottom, left), (top, left)]],
        }
    }
    # db.write_features("polygon_feature", [feat])
    db.write_features("georeference_meta",[feat])
    print('georeference_meta:',feat)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None) 
    parser.add_argument('--temp_dir', type=str, default='temp') 
    parser.add_argument('--support_data_dir', type=str, default='support_data')
    parser.add_argument('--more_info', action='store_true' )
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    assert args.input_path is not None
    assert args.output_path is not None 
    
    seg_bbox, top10, image_width, image_height, title, toponyms = run_georeferencing(args)

    generate_geotiff(args, seg_bbox, top10, image_width, image_height)

    # write_to_geopackage(args, seg_bbox, top1, image_width, image_height)
    write_to_json(args, seg_bbox, top10, image_width, image_height, title, toponyms)

    
if __name__ == '__main__':

    main()

'''
Example command:
to json:
python3 run_georeference.py --input_path='/home/zekun/data/nickel_0209/raw_maps/169_34067.tif' --output_path='debug.json' --support_data_dir='/home/zekun/ta1_georeferencing/geological-map-georeferencing/support_data'

python3 run_georeference_moreinfo.py --input_path='/home/zekun/data/nickel_0209/raw_maps/169_34067.tif' --output_path='/home/zekun/data/nickel_output/0209/169_34067.json'

to geopackage:
python3 run_georeference_moreinfo.py  --input_path='../input_data/CO_Frisco.png' --output_path='../output_georef/CO_Frisco.gpkg'
'''
