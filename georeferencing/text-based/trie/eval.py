from osgeo import gdal
import pandas as pd
import numpy as np
import ast
import pickle

def check_candidate_accuracy(trie_result_df_f, geo_topo_pairs, correct_outpt, incorrect_outpt):
    '''
    trie result dict is the df result of the trie
    geo_topo_pairs is the ground truth dictionary
    '''
    correct = 0
    missing = 0
    correct_dict = {}
    incorrect_map_list = []
    geomaps = trie_result_df_f.geomap_name.tolist()
    topo_cands = trie_result_df_f.topos.tolist()
    for idx, geomap in enumerate(geomaps):
        cand_list = set(topo_cands[idx])
        try:
            #if it's not in here it should be one of the ten missing ones
            topo_gt = set(geo_topo_pairs[geomap])
        except:
            missing += 1
            continue
        if cand_list.intersection(topo_gt):
            correct += 1
            intersection_set = cand_list.intersection(topo_gt)
            correct_dict[geomap] = intersection_set
        else:
            incorrect_map_list.append(geomap)
    
    assert missing == 10   
    df_correct = pd.DataFrame({'map_name': list(correct_dict.keys()),
                   'correct_matches': list(correct_dict.values())})
    print('ACC: ', df_correct.shape[0])
    df_correct.to_csv(correct_outpt)
    with open(incorrect_outpt, 'w') as f:
        for item in incorrect_map_list:
            f.write(item + '\n')
            
with open('/home/yaoyi/shared/critical-maas/testset_100map/gf_map_dict_25OR_DICT.pickle', 'rb') as handle:
    geo_topo_pairs = pickle.load(handle)
geo_topo_pairs = {key.replace('.tif', ''):value for key, value in geo_topo_pairs.items()}  
trie_result_df = pd.read_csv('/home/yaoyi/chen7924/critical-maas/models/trie/outputs/trie_state-quad_100testset_ocr.csv')
trie_result_df['topos'] = trie_result_df['topos'].apply(ast.literal_eval)
trie_result_df_f = trie_result_df[trie_result_df['topos'].apply(lambda x: len(x) > 0)]

correct_output_pt = 'outputs/analysis/trie4merge_correctdict.csv'
incorrect_output_pt = 'outputs/analysis/trie4merge_incorrect.txt'
check_candidate_accuracy(trie_result_df, geo_topo_pairs, correct_output_pt, incorrect_output_pt)