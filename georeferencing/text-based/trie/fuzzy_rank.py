import pandas as pd
import json 
import geopandas as gpd
import numpy as np
import argparse
import ast
import pickle
import pdb 
from fuzzywuzzy import fuzz

def fuzzy_find_top_k_matching(query_title, df, k=10):
    cand_text_list, map_rowid_list = get_topomaps_metadata(df)
    
    matches = []

    for row_id, cand_text in zip(map_rowid_list, cand_text_list):
        title_similarity = fuzz.token_sort_ratio(query_title, cand_text)
        matches.append((row_id, cand_text, title_similarity))

    # Sort the matches based on similarity in descending order
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

    # Return the top k matches
    top_k_matches = sorted_matches[:k]
    return top_k_matches



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

    df_correct = pd.DataFrame({'map_name': list(correct_dict.keys()),
                   'correct_matches': list(correct_dict.values())})
    print('ACC: ', df_correct.shape[0])
    df_correct.to_csv(correct_outpt)
    with open(incorrect_outpt, 'w') as f:
        for item in incorrect_map_list:
            f.write(item + '\n')




def sanity_check(args, trie_result_df): #questionable, how to merge maps
        
    trie_result_df_f

    pass

def fuzzy_top_k(trie_result_df, trie_words_df, topo_meta_df, top_k = 10):

    geomaps = trie_result_df_f.geomap_name.tolist()
    topo_cands = trie_result_df_f.topos.tolist()
    ret_dict = {}

    for idx, geomap in enumerate(geomaps):
        cand_list = list(set(topo_cands[idx]))
        cand_list = sorted(cand_list)
        try:
            #if it's not in here it should be one of the ten missing ones
            topo_gt = set(geo_topo_pairs[geomap])
            
        except:
            print(geomap)
            continue 

        cur_geo_words = trie_words_df[trie_words_df['map_title']==geomap].iloc[0]
        
        query_string = ast.literal_eval(cur_geo_words['state']) + ast.literal_eval(cur_geo_words['quadrangle']) + ast.literal_eval(cur_geo_words['county'])
        query_string = ' '.join(query_string)

        matches = []
        try:
            for idx in range(len(cand_list)):
                topo_cand = cand_list[idx]
                cur_topo_meta = topo_meta_df[topo_meta_df['product_filename']==topo_cand.replace('%20',' ')+'.pdf'].iloc[0]
                cand_string = cur_topo_meta['map_name'] + cur_topo_meta['primary_state'] +  cur_topo_meta['county_list']
                similarity = fuzz.token_sort_ratio(query_string, cand_string)
                matches.append((idx, topo_cand, similarity))
        except:
            pdb.set_trace()
        
        sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)

        # Return the top k matches
        ret_dict[geomap] = {}
        top_ks = [1,2,5,10]
        for top_k in top_ks:
            top_k_matches = sorted_matches[:top_k]
            ret_dict[geomap]['topos_top'+str(top_k)] = [a[1] for a in top_k_matches]
            ret_dict[geomap]['topo_gt'] = topo_gt

        
    return ret_dict
        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--geo2topo_gt_path', type=str, default='/home/yaoyi/shared/critical-maas/testset_100map/gf_map_dict_25OR_DICT.pickle')
    parser.add_argument('--trie_result_path', type=str, default='/home/yaoyi/shared/critical-maas/tree-crit-maas-code/models/trie/outputs/trie3_state-quad-county_100testset_ocr.csv') 
    parser.add_argument('--trie_words_path', type=str, default='/home/yaoyi/shared/critical-maas/tree-crit-maas-code/models/trie/outputs/MATCHES_trie_state-quad-county_100testset_ocr.csv')
    parser.add_argument('--topo_meta_path', type=str, default='/home/yaoyi/shared/critical-maas/tree-crit-maas-code/Data/usgs_topo_geoms.geojson')
    
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')


    with open(args.geo2topo_gt_path, 'rb') as handle:
        geo_topo_pairs = pickle.load(handle)

    geo_topo_pairs = {key.replace('.tif', ''):value for key, value in geo_topo_pairs.items()}  

    trie_words_df = pd.read_csv(args.trie_words_path)

    topo_meta_df = gpd.read_file(args.topo_meta_path)

    trie_result_df = pd.read_csv(args.trie_result_path)
    trie_result_df['topos'] = trie_result_df['topos'].apply(ast.literal_eval)
    trie_result_df_f = trie_result_df[trie_result_df['topos'].apply(lambda x: len(x) > 0)]

    # sanity_check(args, trie_result_df)

    fuzzy_dict = fuzzy_top_k(trie_result_df, trie_words_df, topo_meta_df)

    with open('fuzzy_dict.json','w') as f:
        json.dump(fuzzy_dict, f)

    fuzzy_df = pd.DataFrame.from_dict(fuzzy_dict).transpose()
    fuzzy_df = fuzzy_df.rename(columns={0: 'geomap_name'})

    pdb.set_trace()
    check_fuzzy_accuracy(fuzzy_df, geo_topo_pairs)

    # correct_output_pt = 'outputs/analysis/trie4merge_correctdict.csv'
    # incorrect_output_pt = 'outputs/analysis/trie4merge_incorrect.txt'
    # check_candidate_accuracy(trie_result_df, geo_topo_pairs, correct_output_pt, incorrect_output_pt)

    # check_candidate_accuracy(fuzzy_df, geo_topo_pairs, correct_output_pt, incorrect_output_pt)