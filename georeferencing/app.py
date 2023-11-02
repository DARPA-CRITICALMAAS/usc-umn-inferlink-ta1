import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import cv2
from utils import get_toponym_tokens, prepare_bm25
from segmentation_sam import resize_img, run_sam
from streamlit_folium import folium_static
import folium
import rasterio
import numpy as np
import torch
import pdb 


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



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
    # image = Image.open(file) # read image with PIL library


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




# main title
st.title("Geologic Map Auto Georeferencing") 
# subtitle
st.markdown("### Upload Map")



file = st.file_uploader(label = "Upload your map", type=['png', 'jpg', 'jpeg', '.tif'])


if file:
    seg_mask, seg_bbox, image = run_segmentation(file)
    print(seg_bbox)


st.markdown("### Enter Title and Basemap Description")



title = st.text_input('Map title', '')
st.write('\t Example: Preliminary Materials Map Plainfield Quadrangle Massachusetts.')
# st.write('\t Example: Geologic Map of The Lake Helen Quadrangle, Big Horn and Johnson Counties, Wyoming.')


basemap_descrip = st.text_input('Basemap Description', '')
st.write('\t Example: Base map by the United States Geological Survey, 1955')
# st.write('\t Example: Base from U.S. Geological Survey, 1967')

query_sentence = title + ' ' + basemap_descrip 
st.write('Query: ', query_sentence)


bm25, df_merged = load_data(topo_histo_meta_path = 'support_data/historicaltopo.csv',
    topo_current_meta_path = 'support_data/ustopo_current.csv')


# def click_first_button():
#     st.session_state.first.clicked = True

# def click_second_button():
#     st.session_state.second.clicked = True

w1 = st.button("Start Processing")

if st.session_state.get('w1') != True:

    st.session_state['w1'] = w1


if st.session_state['w1'] == True:

    query_tokens, human_readable_tokens = get_toponym_tokens(query_sentence, device)


    st.write('Detected toponyms:', human_readable_tokens)

    query_sent = ' '.join(human_readable_tokens)

    # w2 = st.button("Find Base Topo Map") 

    # if st.session_state.get('w2') != True:

    #     st.session_state['w2'] = w2

    
    # if st.session_state['w2'] == True:
        

    tokenized_query = query_sent.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    sorted_bm25_list = np.argsort(doc_scores)[::-1]


    st.write("Top retrievals from the topo map database.")
    
    st.dataframe(df_merged.iloc[sorted_bm25_list[0:10]])


    top1 = df_merged.iloc[sorted_bm25_list[0]]

    st.markdown("### Display Georeferencing Result")

    folium_plot_map(file, seg_bbox, top1)
    
    