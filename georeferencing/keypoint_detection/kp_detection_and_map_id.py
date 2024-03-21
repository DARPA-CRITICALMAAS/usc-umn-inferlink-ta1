import sys
sys.path.append('dino-vit-features')
from correspondences import draw_correspondences, find_correspondences
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import ast
import json
import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def return_gcp_from_geomap(geomap_pt, topomap_pts, topomap_root):
    #@markdown Choose number of points to output:
    num_pairs = 5 #@param
    #@markdown Choose loading size:
    load_size = 128 #@param
    #@markdown Choose layer of descriptor:
    layer = 9 #@param
    #@markdown Choose facet of descriptor:
    facet = 'key' #@param
    #@markdown Choose if to use a binned descriptor:
    bin=True #@param
    #@markdown Choose fg / bg threshold:
    thresh=0.05 #@param #might have to alter this
    #@markdown Choose model type:
    model_type='dino_vits8' #@param
    #@markdown Choose stride:
    stride=8 #@param
    
    topo_folder = topomap_pts
    max_sim_tracker = float('-inf')
    max_topo = ''
    for topo in topo_folder:
        abs_pt = os.path.join(topomap_root, topo +'.tif')
        with torch.no_grad():
            sims, points1, points2, image1_pil, image2_pil = find_correspondences(geomap_pt, abs_pt, num_pairs, load_size, layer,
                                                                   facet, bin, thresh, model_type, stride)
            overall_sim = np.sum(sims)
            if overall_sim > max_sim_tracker:
                max_sim_tracker = overall_sim
                max_sims = sims
                max_points1 = points1
                max_points2 = points2
                max_image1_pil = image1_pil
                max_image2_pil = image2_pil
                max_topo = topo
    return max_sims, max_points1, max_points2, image1_pil, image2_pil, max_topo
          
def save_keypoints_as_json(geomap_coords, topomap_coords, json_outpt):
    keypoints_data = []
    for i, (coord1, coord2) in enumerate(zip(geomap_coords, topomap_coords)):
        keypoint_entry = {
            'id': i,
            'geomap_coords': {'x': coord1[0], 'y': coord1[1]},
            'topomap_coords': {'x': coord2[0], 'y': coord2[1]}
        }
        keypoints_data.append(keypoint_entry)
    with open(json_outpt, 'w') as json_file:
        json.dump(keypoints_data, json_file, indent=4)
    print(f"JSON file '{json_outpt}' with interpolated coordinates has been created successfully.")
    
def interpolate_coords(resized_size, original_size, coord):
    scale_x = original_size[0] / resized_size[0]
    scale_y = original_size[1] / resized_size[1]
    
    return int(coord[0] * scale_x), int(coord[1] * scale_y)

def main(geomap_folder, topomap_folder, pairs_df_pt):
    pairs_df = pd.read_csv(pairs_df_pt)
    
    pairs_df['topos'] = pairs_df['topos'].apply(ast.literal_eval)
    pairs_df = pairs_df[pairs_df['topos'].apply(lambda x: len(x) > 0)]
    
    geomaps = pairs_df.geomap_name.tolist()
    total_sims = []
    total_topos = []
    geomap_fps = os.listdir(geomap_folder)
    for idx, geomap in enumerate(geomap_fps):
        topomap_list = pairs_df[pairs_df['geomap_name'] == geomap.replace('.tif', '')].topos.tolist()[0]
        abs_geomap_pt = os.path.join(geomap_folder, geomap)
        sims, points_geomap, points_topomap, geomap_pil, topomap_pil, topo = return_gcp_from_geomap(abs_geomap_pt, topomap_list, topomap_folder)
        original_size_geomap = Image.open(abs_geomap_pt).size
        topomap_pt = os.path.join(topomap_folder, topo +'.tif')
        original_size_topomap = Image.open(topomap_pt).size

        resized_size_geomap = geomap_pil.size
        resized_size_topomap = topomap_pil.size

        geomap_points = [interpolate_coords(resized_size_geomap, original_size_geomap, coord) for coord in points_geomap]
        topomap_points =[interpolate_coords(resized_size_topomap, original_size_topomap, coord) for coord in points_topomap]

        json_outpt = 'outputs/' + geomap +'.json'
        save_keypoints_as_json(geomap_points, topomap_points, json_outpt)
        total_sims.append(sims)
        total_topos.append(topo)
    results = {'geomap_name': geomaps,
        'topo_name': total_topos,
        'sims': total_sims}
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/kp_detect_results_ocr1.csv')
    
if __name__ == "__main__":
    geomap_folder = '/home/yaoyi/shared/critical-maas/testset_100map/results/03112024/ided_geomaps'
    topomap_folder = '/home/yaoyi/shared/critical-maas/testset_100map/results/03112024/ided_topomaps_2'
    pairs_df_pt = '/home/yaoyi/chen7924/critical-maas/models/trie/outputs/trie_state-quad-county_100testset_ocr.csv'
    main(geomap_folder, topomap_folder, pairs_df_pt)
        
    