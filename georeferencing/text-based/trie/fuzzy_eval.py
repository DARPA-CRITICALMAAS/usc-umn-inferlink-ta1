import json 
import pandas as pd
import argparse
import pickle

def get_accuracy(geomaps, topo_cands, geo_topo_pairs):
    correct = 0
    missing = 0
    correct_dict = {}
    incorrect_map_list = []

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
    # df_correct.to_csv(correct_output)
    # with open(incorrect_outpt, 'w') as f:
    #     for item in incorrect_map_list:
    #         f.write(item + '\n')


def check_fuzzy_accuracy(fuzzy_dict, geo_topo_pairs):
    '''
    trie result dict is the df result of the trie
    geo_topo_pairs is the ground truth dictionary
    '''
   
    # geomaps = fuzzy_df.geomap_name.tolist()
    geomaps = list(fuzzy_dict.keys())

    

    top_ks = [1,2,5,10]
    for top_k in top_ks:

        # topk_cands = fuzzy_dict['topos_top'+str(top_k)].tolist()
        topk_cands = [fuzzy_dict[geomap]['topos_top'+str(top_k)] for geomap in fuzzy_dict]

        get_accuracy(geomaps, topk_cands, geo_topo_pairs)
        





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--geo2topo_gt_path', type=str, default='/home/yaoyi/shared/critical-maas/testset_100map/gf_map_dict_25OR_DICT.pickle')
    
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    with open(args.geo2topo_gt_path, 'rb') as handle:
        geo_topo_pairs = pickle.load(handle)

    geo_topo_pairs = {key.replace('.tif', ''):value for key, value in geo_topo_pairs.items()}  
    

    with open('fuzzy_dict.json','r') as f:
        fuzzy_dict = json.load(f)

    check_fuzzy_accuracy(fuzzy_dict, geo_topo_pairs)