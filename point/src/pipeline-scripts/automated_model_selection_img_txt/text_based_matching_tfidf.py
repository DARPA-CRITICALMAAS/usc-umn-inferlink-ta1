### 
# This script is mainly working for high pirority symbols
###

import json
import re
import os
import argparse
import sklearn
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_string(s):
    new_s = re.sub(r"\s+", ' ', s)
    new_s = new_s.lower()
    new_s = new_s.replace('—', ' ')
    new_s = new_s.replace('-', ' ')
    return new_s

def replace_synonym(s):
    s = s.replace('schistosity', 'foliation')
    s = s.replace('plunge', 'lineation')
    return s
    
    
# we predefine some rules for the lines
def is_potential_line_feature(description):
    line_inits = ["solid line", "dashed line", "dotted line", "serrated line", "long", "syncline", "fault", "anticline"]
    for line_init in line_inits:
        if description.startswith(line_init):
            return True
    return False
    
# we predefine some rules for the points
def is_potential_pt_feature(description):
    if "strike and dip of beds" in description and "overturned" not in description and "foliation" not in description:
        return "inclined_bedding"
    elif "strike and dip of foliation" in description:
        return "inclined_metamorphic"
    elif "overturned" in description:
        return "overturned_bedding"
    elif "quarry" in description: # this is hard coded need to use image based approach
        return "quarry"
    else:
        return None
        
def update_title(description, title):
    if "strike and dip" in description and "bed" in description:
        return "strike and dip of beds"
    elif "strike and dip" in description and "foliation" in description:
        return "strike and dip of foliation"
    else:
        return title

def add_title_to_description(description, title):
    point_inits = ["inclined", "vertical", "overturned", "horizontal"]
    # case 1
    if description in point_inits:
        return title + " " + description
    # case 2    
    for point_init in point_inits:
        # print(description)
        if description.startswith(point_init) and "title" not in point_init:
            return title + " " + description
    return description
    
    
def preprocess_gpt_metadata(metadata):
    selected_models = []
    remain_descriptions = []

    cur_title = ""
    for k, v in metadata.items():
        if type(v) == dict:
            symbol_name = clean_string(v['symbol name'])
            if is_potential_line_feature(symbol_name):
                continue

            descriptions = v['description'].split(';')
            for description in descriptions:
                description = clean_string(description).lower()
                cur_title = update_title(symbol_name + ' ' + description, cur_title)
                new_description = add_title_to_description(description, cur_title)
                full_description = symbol_name + '####' + new_description
                full_description = replace_synonym(full_description)

                model = is_potential_pt_feature(full_description)
                if model is not None:
                    selected_models.append(model)
                else:
                    remain_descriptions.append(full_description)   
    return remain_descriptions, list(set(selected_models))


def preprocess_layout_metadata(metadata):
    selected_models = []
    remain_descriptions = []

    for k, v in metadata.items():        
        descriptions = v.split(';')
        for description in descriptions:
            desc = clean_string(description).lower()
            if is_potential_line_feature(desc): 
                continue
                
            desc = replace_synonym(desc)
            model = is_potential_pt_feature(desc)
            if model is not None:
                selected_models.append(model)
            else:
                remain_descriptions.append('####' + desc)   
    return remain_descriptions, list(set(selected_models))



def find_category(description, categories):
    vectorizer = TfidfVectorizer()
    scores = dict()
    for cat in categories:
        vectors = vectorizer.fit_transform([description, cat])
        scores[cat] = cosine_similarity(vectors)[0, 1]
        
    selected_cat, score = [(k, v) for k, v in sorted(scores.items(), key=lambda item: -item[1])][0]
    if score > 0:
        return selected_cat
    else:
        return None
    
    
def text_based_matching(input_map_name, 
                        metadata_path, 
                        metadata_type,
                        symbol_info_json_file="symbol_info.json", 
                        use_shape=True, 
                        use_keywords=True, 
                        use_long_description=False):
 
    use_shape = metadata_type == 'gpt'
    
    if not os.path.exists(symbol_info_json_file):
        return []
    
    with open(symbol_info_json_file, 'r') as f:
        symbol_info = json.load(f)
        symbol_desc_dict = symbol_info['symbol_description']
        symbol_cat_dict = symbol_info['symbol_category']
        symbol_categories = list(symbol_cat_dict.keys())
        
    # initialize 
    map_selected_models = dict()
    vectorizer = TfidfVectorizer()
    
    if metadata_type == 'gpt':
        metadata_file = os.path.join(metadata_path, input_map_name + '_gpt_point.json')
    elif  metadata_type == 'layout':
        metadata_file = os.path.join(metadata_path, input_map_name + 'gpt_point.json')
    
    if not os.path.exists(metadata_file):
        print('not exist')
        return []
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
        
    if metadata_type == 'gpt':
        
        remain_descriptions, selected_models = preprocess_gpt_metadata(metadata)   

    elif metadata_type == 'layout':
        remain_descriptions, selected_models = preprocess_layout_metadata(metadata)     
    
    for description in remain_descriptions:
        tar_shape_des, tar_sym_des = description.split('####')
        if tar_sym_des.isspace(): tar_sym_des = tar_shape_des
                    
        tar_cat = find_category(description, symbol_categories)
        if tar_cat is None: continue; # cannot find category
        if len(symbol_cat_dict[tar_cat]) == 0: continue; # no candidate symbols
        
        scores = dict()
        for sym in symbol_cat_dict[tar_cat]:
            shape2 = symbol_desc_dict[sym]['shape']
            keyword2 = symbol_desc_dict[sym]['keywords']
            des2 = symbol_desc_dict[sym]['description']    
                        
            sim_shape, sim_kws, sim_des = 0., 0., 0.
            if use_shape or len(tar_shape_des) == 0:
                vectors = vectorizer.fit_transform([tar_shape_des, shape2])
                sim_shape = cosine_similarity(vectors)[0, 1]
            if use_keywords:
                vectors = vectorizer.fit_transform([tar_sym_des, keyword2])
                sim_kws = cosine_similarity(vectors)[0, 1]
            if use_long_description:    
                vectors = vectorizer.fit_transform([tar_sym_des, des2])
                sim_des = cosine_similarity(vectors)[0, 1]

            scores[sym] = sim_shape + sim_kws + sim_des
                        
        selected_model, score = [(k, v) for k, v in sorted(scores.items(), key=lambda item: -item[1])][0]
        selected_models.append(selected_model)
                
    return list(set(selected_models))



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='symbol')
#     parser.add_argument('--symbol_description_json_file', type=str, default="symbol_info.json")
#     parser.add_argument('--metadata_path', type=str, default="../point_metadata_nickel")
#     parser.add_argument('--metadata_type', type=str, default="gpt")
#     args = parser.parse_args()
    
#     all_maps = glob.glob("/home/yaoyi/shared/critical-maas/ta1_feature_extraction/evaluation/*.tif")
#     all_maps = sorted([os.path.basename(m) for m in all_maps])
    
#     out = {}
#     for map_filename in all_maps:
#         map_name = map_filename.replace('.tif', '')
#         map_selected_models = text_based_matching(map_name, 
#                                                   args.metadata_path, 
#                                                   args.metadata_type, 
#                                                   symbol_info_json_file=args.symbol_description_json_file,
#                                                   use_shape=args.metadata_type=='gpt', 
#                                                   use_keywords=True, use_long_description=False)
#         out[map_name] = map_selected_models
        
#     with open(f'../{metadata_type}_selection.json', 'w') as f:
#         json.dump(out, f)
    
#     # python text_based_matching_tfidf.py --metadata_path ../legend_item_data/gpt_evaluation_set/legend_item_description_outputs/evaluation --metadata_type gpt
#     # python text_based_matching_tfidf.py --metadata_path ../legend_item_data/layout_evaluation_set/ --metadata_type layout