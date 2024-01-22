import pandas as pd
import numpy as np
from PIL import Image
import cv2
from utils import get_toponym_tokens, prepare_bm25
from segmentation_sam import resize_img, run_sam
# from criticalmaas.ta1_geopackage import GeopackageDatabase
import json 
import numpy as np
import torch
import base64
import requests
import argparse
import io
import os



Image.MAX_IMAGE_PIXELS = None

# torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


api_key = os.getenv("OPENAI_API_KEY")



def run_segmentation(file):
    image = Image.open(file) # read image with PIL library

    print('image_size',image.size)

    resized_img, scaling_factor = resize_img(np.array(image))
    seg_mask, bbox = run_sam(np.array(image), resized_img, scaling_factor, device)

    return seg_mask, bbox, image, image.width, image.height


def load_data(topo_histo_meta_path, topo_current_meta_path):

    # topo_histo_meta_path = 'support_data/historicaltopo.csv'
    df_histo = pd.read_csv(topo_histo_meta_path) 
            
    # topo_current_meta_path = 'support_data/ustopo_current.csv'
    df_current = pd.read_csv(topo_current_meta_path) 


    common_columns = df_current.columns.intersection(df_histo.columns)

    df_merged = pd.concat([df_histo[common_columns], df_current[common_columns]], axis=0)
            
    bm25 = prepare_bm25(df_merged)

    return bm25, df_merged


def folium_plot_map(file, seg_bbox, top1):
    image = Image.open(file) # read image with PIL library


    left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
    bounds = [(bottom, left), (top, right)]


    # st.write("Plotting maps...")

    m = folium.Map(location=[0.5*(top+bottom), 0.5*(left+right)], zoom_start=10)

    # # add marker for Liberty Bell
    # tooltip = "Manilla city"
    # folium.Marker(
    #     [14.599512, 120.984222], popup="This is it!", tooltip=tooltip
    # ).add_to(m)


    img = folium.raster_layers.ImageOverlay(
        name="Geologic Map",
        image=np.array(image)[seg_bbox[1]:seg_bbox[1]+seg_bbox[3],seg_bbox[0]:seg_bbox[0]+seg_bbox[2],:],
        bounds=bounds,
        opacity=0.9,
        interactive=True,
        cross_origin=False,
        zindex=1,
    )

    img.add_to(m)
    folium.LayerControl().add_to(m)

    # call to render Folium map in Streamlit
    folium_static(m)


def get_topo_basemap(query_sentence, bm25, df_merged, device ):

    print(query_sentence)
    query_tokens, human_readable_tokens = get_toponym_tokens(query_sentence, device)

    query_sent = ' '.join(human_readable_tokens)

    tokenized_query = query_sent.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    sorted_bm25_list = np.argsort(doc_scores)[::-1]
    
    # Top 10
    # df_merged.iloc[sorted_bm25_list[0:10]]

    top1 = df_merged.iloc[sorted_bm25_list[0]]

    return top1 


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


#check and downscale:
def downscale(image, max_size=13):
    
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")  
    img_size = buffer.tell() / (1024 * 1024)
    
    if img_size > max_size:
        downscale_factor = max_size / img_size
        downscale_factor = max(downscale_factor, 0.1)
        
        new_size = (int(image.width * downscale_factor), int(image.height * downscale_factor))
        
        downscaled_img = image.resize(new_size)
        
        # st.write(f"image has downscaled")
        # st.image(downscaled_img)
        
        return downscaled_img
    else:
        return image

def to_camel(title):
    words = title.split()
    return ' '.join(word.capitalize() for word in words)


def run_georeferencing(args):

    # input_path = '../input_data/CO_Frisco.png' # supported file format: png, jpg, jpeg, tif
    input_path = args.input_path

    # load data store in bm25 & df_merged
    bm25, df_merged = load_data(topo_histo_meta_path = 'support_data/historicaltopo.csv',
        topo_current_meta_path = 'support_data/ustopo_current.csv')
    
    seg_mask, seg_bbox, image, image_width, image_height = run_segmentation(input_path)

    if not os.path.isdir('temp'):
        os.makedirs('temp')
    
    jpg_file_path = "temp/output.jpg"

    image.save(jpg_file_path, format="JPEG")

    image =downscale(image)
    # Getting the base64 string
    base64_image = encode_image(image)
    title = to_camel(getTitle(base64_image))

    os.remove("temp/output.jpg")

    query_sentence = title

    # get the highest score: 
    top1 = get_topo_basemap(query_sentence, bm25, df_merged, device )

    return seg_bbox, top1, image_width, image_height

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

def write_to_json(args, seg_bbox, top1, width, height):
    left, right, top, bottom = top1['westbc'], top1['eastbc'], top1['northbc'], top1['southbc']
    img_left, img_right, img_top, img_bottom = seg_bbox[0], seg_bbox[0]+seg_bbox[2], seg_bbox[1], seg_bbox[1]+seg_bbox[3]
    
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
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    assert args.input_path is not None
    assert args.output_path is not None 
    
    seg_bbox, top1, image_width, image_height = run_georeferencing(args)

    # write_to_geopackage(args, seg_bbox, top1, image_width, image_height)
    write_to_json(args, seg_bbox, top1, image_width, image_height)

    


if __name__ == '__main__':

    main()

'''
Example command:
to json:
python3 run_georenference.py  --input_path='../input_data/CO_Frisco.png' --output_path='../output_georef/CO_Frisco.json'

to geopackage:
python3 run_georenference.py  --input_path='../input_data/CO_Frisco.png' --output_path='../output_georef/CO_Frisco.gpkg'
'''