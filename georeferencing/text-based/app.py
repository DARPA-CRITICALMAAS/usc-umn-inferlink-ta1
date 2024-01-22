import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import cv2
from utils import get_toponym_tokens, prepare_bm25
from segmentation_sam import resize_img, run_sam
from streamlit_folium import folium_static
import folium
import numpy as np
import torch
import base64
import requests
import io
from transformers import pipeline
import os



Image.MAX_IMAGE_PIXELS = None

# torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



api_key = os.getenv("OPENAI_API_KEY")


def overlay_mask(input_img, mask_img, transparency):

    mask_img = np.expand_dims(mask_img, -1)
    zeros = np.zeros_like(mask_img)
    mask_img = np.concatenate((zeros, mask_img, zeros), axis=-1)
    combined_img = cv2.addWeighted(input_img,0.6,mask_img,0.4,0)

    return combined_img

@st.cache_data
def run_segmentation(file):
    image = Image.open(file) # read image with PIL library

    print(image.size)

    st.write('Original Image:')
    st.image(image) #display

    resized_img, scaling_factor = resize_img(np.array(image))
    seg_mask, bbox = run_sam(np.array(image), resized_img, scaling_factor, device)


    overlay_img = overlay_mask(np.array(image), seg_mask, transparency = 0.5)

    st.write('Segmentation Mask:')
    st.image(overlay_img)

    return seg_mask, bbox, image


@st.cache_data
def load_data(topo_histo_meta_path, topo_current_meta_path):

    # topo_histo_meta_path = 'support_data/historicaltopo.csv'
    df_histo = pd.read_csv(topo_histo_meta_path) 
            
    # topo_current_meta_path = 'support_data/ustopo_current.csv'
    df_current = pd.read_csv(topo_current_meta_path) 


    common_columns = df_current.columns.intersection(df_histo.columns)

    df_merged = pd.concat([df_histo[common_columns], df_current[common_columns]], axis=0)
            
    bm25 = prepare_bm25(df_merged)

    return bm25, df_merged

@st.cache_data
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

@st.cache_data
def run_georeference(query_sentence, device ):

    print(query_sentence)
    query_tokens, human_readable_tokens = get_toponym_tokens(query_sentence, device)

    st.write('Detected toponyms:', human_readable_tokens)

    query_sent = ' '.join(human_readable_tokens)

    tokenized_query = query_sent.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    sorted_bm25_list = np.argsort(doc_scores)[::-1]


    st.write("Top retrievals from the topo map database.")
    
    st.dataframe(df_merged.iloc[sorted_bm25_list[0:10]])


    top1 = df_merged.iloc[sorted_bm25_list[0]]

    return top1 



# Function to encode the image
def encode_image(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return encoded_image

@st.cache_data
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
    
    try:
        title = response.json()['choices'][0]['message']['content']
    except:
        print(response)
        exit(-1)

    return title 


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

        
# def getScale(base64_image):
#     if base64_image is None:
#         return "No file selected"
    
    
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }
    
#     payload = {
#         "model": "gpt-4-vision-preview",
#         "messages": [
#             {
#             "role": "user",
#             "content": [
#               {
#                 "type": "text",
#                 "text": "What’s the Scale of map? please just return the title no more words else"
#               },
#               {
#                 "type": "image_url",
#                 "image_url": {
#                   "url": f"data:image/jpeg;base64,{base64_image}"
#                 }
#               }
#             ]
#             }
#         ],
#         "max_tokens": 300
#     }

#     response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
#     return response.json()['choices'][0]['message']['content']

    
# main title
st.title("Geologic Map Auto Georeferencing") 
# subtitle
st.markdown("### Step 1: Upload Map")


file = st.file_uploader(label = "Upload your map", type=['png', 'jpg', 'jpeg', '.tif'])


if file:
    seg_mask, seg_bbox, image = run_segmentation(file)

    print(seg_bbox)



# load data store in bm25 & df_merged
bm25, df_merged = load_data(topo_histo_meta_path = 'support_data/historicaltopo.csv',
    topo_current_meta_path = 'support_data/ustopo_current.csv')



# trans Tif. to base64

if file:
    st.markdown("### Step 2: Automaticly extract title from map")

    # create a tempory folder to store the jpg
    if not os.path.isdir('temp'):
        os.makedirs('temp')
      
    jpg_file_path = "temp/output.jpg"
    
    image.save(jpg_file_path, format="JPEG")
  

    image =downscale(image)
    
    # Getting the base64 string
    base64_image = encode_image(image)
    

    # title = to_camel(getTitle(base64_image))
    if 'title' not in st.session_state:
        st.session_state.title = to_camel(getTitle(base64_image))

    modefied = st.text_area("Title of the map:", st.session_state.title)
    


    os.remove("temp/output.jpg")


    w1 = st.button("Start Processing")
    
    if w1:

        query_sentence = modefied
        st.write('Query: ', query_sentence)

        # get the highest score: 
        top1 = run_georeference(query_sentence, device )
        
        st.markdown("### Display Georeferencing Result")
        
        #show on the map: 
        folium_plot_map(file, seg_bbox, top1)

        # torch.cuda.empty_cache()
        del st.session_state.title
