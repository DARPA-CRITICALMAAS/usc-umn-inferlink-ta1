import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
import pickle
import sys
import argparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from correspondences import find_sims
import multiprocessing
from tqdm import tqdm
from functools import partial

def calc_geo_topo_correlation(gt_pair, gt_true_dict, args):# Paths to your images
    geo_name, topos = gt_pair
    geo_root = args.geo_root
    topo_root = args.topo_root
    geomap_path = os.path.join(geo_root, geo_name +'.tif')
    
    #For each geo-topo pair calculate, bbp sum similarity
    sim_scores = []
    for topo in topos:
        topomap_path = os.path.join(topo_root, topo + '.tif')
        sim = find_sims(geomap_path, topomap_path)
        sim_scores.append(sim)
    
    combined = list(zip(sim_scores, topos))
    sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)

    # Extract the sorted lists of integers and strings
    sorted_integers, sorted_strings = zip(*sorted_combined_list)
    
    print(sorted_strings[0])
    print(gt_true_dict[geo_name])
    sys.exit()
    if sorted_strings[0] in gt_true_dict[geo_name]:
        return 1, (geo_name, sorted_strings)
    else:
        return 0, (geo_name, sorted_strings)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--geo_root', type=str, default='/scratch.global/chen7924/cropped_NGMDB_GF',
                        help='root dir geo files')
    parser.add_argument('--topo_root', type=str, default='/scratch.global/chen7924/topomaps_0101/topomaps_1130',
                        help='dir of topo files')
    parser.add_argument('--cropped', action='store_true', default=False)
    parser.add_argument('--out_list', type=str, default='/scratch.global/chen7924/semcorr_results/dino_vit_pairs.pkl',
                        help='dir of topo files')
    cmd_args = parser.parse_args()

    #load in dict with main pairs
    with open('/home/yaoyi/chen7924/critical-maas/models/vit-topo/data/mn_geo-topo_pairs_90OR90.pickle', 'rb') as handle:
        geo_topo_pairs = pickle.load(handle)
        
    #Load in dict with incorrect rank 20 pairs
    with open('/home/yaoyi/chen7924/critical-maas/Data/semantic_correspondence/sem_corr_1-1_top20_f.pkl', 'rb') as handle:
        rank20_dict_corr = pickle.load(handle)
    
    rank20_dict = {}
    for geo_name in rank20_dict_corr.keys():
        geomap_path = os.path.join(cmd_args.geo_root, geo_name +'.tif')
        if os.path.exists(geomap_path):
            rank20_dict[geo_name] = rank20_dict_corr[geo_name]
    print(len(list(rank20_dict.keys())))
    # Function to save strings to a pickle file
    def save_strings(strings):
        with open(cmd_args.out_list, "wb") as file:
            pickle.dump(strings, file)   
            
    results_list = []
    strings_list = []
    loop_counter = 0
    for key, value in rank20_dict.items():
        result, pairs = calc_geo_topo_correlation((key, value), geo_topo_pairs, cmd_args)
        results_list.append(result)
        strings_list.append(value)
        loop_counter += 1
        if loop_counter % 100 == 0:
            print(f"Accuracy after {loop_counter} threads: {sum(results_list)/len(results_list)}")
            # Save the strings every 100 threads
            save_strings(strings_list)
    
    save_strings(strings_list)
    print(f"Accuracy after {loop_counter} threads: {sum(results_list)/len(results_list)}")
        
        
#     num_threads = 1
#     partial_sim = partial(calc_geo_topo_correlation, gt_true_dict = geo_topo_pairs, args = cmd_args)
#     results_list = []
#     strings_list = []
#     pool = multiprocessing.Pool(processes = num_threads)
#     with tqdm(total=len(list(rank20_dict.keys())), desc="Processing", unit="key") as pbar:
#         # Use imap_unordered to parallelize the function execution and get the results
#         for result in pool.imap_unordered(partial_sim, rank20_dict.items()):
#             pbar.update()
#             results_list.append(result[0])
#             strings_list.append(result[1])

#             if pbar.n % 100 == 0:
#                 print(f"Sum after {pbar.n} threads: {sum(results_list)}")
#                 # Save the strings every 100 threads
#                 save_strings(strings_list)

 
#     pool.close()
#     pool.join()
#     total_sum = sum(results)