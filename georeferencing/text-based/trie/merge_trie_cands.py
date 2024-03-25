import pandas as pd
import geopandas as gpd
import numpy as np
import ast
import pickle

def merge_cands(cand_list):
    new_cand_list = []
    seen_maps = []
    for cand in cand_list:
        clean_cand = cand.split('_')[0] + cand.split('_')[1]
        if clean_cand not in seen_maps:
            seen_maps.append(clean_cand)
            new_cand_list.append(cand)
        else:
            continue
            
    return new_cand_list

trie_result_df = pd.read_csv('outputs/trie_state-quad-county_100testset_ocr.csv')
trie_result_df['topos'] = trie_result_df['topos'].apply(ast.literal_eval)
# trie_result_df_f = trie_result_df[trie_result_df['topos'].apply(lambda x: len(x) > 0)]

# Apply the function to each item in the 'topos' column
trie_result_df['topos'] = trie_result_df['topos'].apply(merge_cands)

# Add a new column 'topo_count' containing the number of items in each list
trie_result_df['topo_count'] = trie_result_df['topos'].apply(len)

trie_result_df.to_csv('outputs/analysis/trie3_cand_count.csv')