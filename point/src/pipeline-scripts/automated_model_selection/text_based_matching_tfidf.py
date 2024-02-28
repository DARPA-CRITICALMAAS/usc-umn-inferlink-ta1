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
    elif "location" in description and "abandoned" in description and "quarry" in description: # this is hard coded need to use image based approach
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
    
    
def preprocess_metadata(metadata):
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
                full_description = symbol_name + ' - ' + new_description
                full_description = replace_synonym(full_description)

                model = is_potential_pt_feature(full_description)
                if model is not None:
                    selected_models.append(model)
                else:
                    remain_descriptions.append(full_description)   
    return remain_descriptions, selected_models


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
    
    
def text_based_matching(input_map_name, metadata_path, 
                        symbol_info_json_file="symbol_info.json", 
                        use_shape=True, 
                        use_keywords=True, 
                        use_long_description=False):
 
    with open(symbol_info_json_file, 'r') as f:
        symbol_info = json.load(f)
        symbol_desc_dict = symbol_info['symbol_description']
        symbol_cat_dict = symbol_info['symbol_category']
        symbol_categories = list(symbol_cat_dict.keys())
        
    # initialize 
    map_selected_models = dict()
    vectorizer = TfidfVectorizer()
    
    for metadata_file in glob.glob(os.path.join(metadata_path, "*.json")):                                       
        map_name = os.path.basename(metadata_file)
        map_name = map_name.replace('_point.json','')
        
        if input_map_name in map_name:
            if map_selected_models.get(map_name) is None:
                map_selected_models[map_name] = []
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            remain_descriptions, selected_models = preprocess_metadata(metadata)        
            for description in remain_descriptions:

                tar_shape_des, tar_sym_des = description.split('-')
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
                    if use_shape:
                        vectors = vectorizer.fit_transform([tar_shape_des, shape2])
                        sim_shape = cosine_similarity(vectors)[0, 1]
                    if use_keywords:
                        vectors = vectorizer.fit_transform([tar_sym_des, keyword2])
                        sim_kws = cosine_similarity(vectors)[0, 1]
                    if use_long_description:    
                        vectors = vectorizer.fit_transform([tar_sym_des, des2])
                        sim_des = cosine_similarity(vectors)[0, 1]

                    scores[sym] = sim_shape + sim_kws + sim_des
                    
                #     print(tar_cat, "--", tar_shape_des, "|", tar_sym_des, "|", keyword2, "|", des2)
                # print(scores)
                selected_model, score = [(k, v) for k, v in sorted(scores.items(), key=lambda item: -item[1])][0]
                # print("===>", selected_model, score, '\n')
                if selected_model in selected_models:
                    continue
                else:
                    selected_models.append(selected_model)
            
            map_selected_models[map_name] += selected_models

        
    new_map_selected_models = dict()
    for map_name, selected_models in map_selected_models.items():
        new_selected_models = list(set(selected_models))
        new_selected_models = [m + '.pt' for m in new_selected_models]
        new_map_selected_models[map_name] = new_selected_models

    return new_map_selected_models[input_map_name]


if __name__ == "__main__":
    
    # basic parameters
    parser = argparse.ArgumentParser(description='symbol')
    parser.add_argument('--symbol_description_json_file', type=str, default="symbol_info.json")
    parser.add_argument('--metadata_path', type=str, default="../point_metadata_nickel")
    parser.add_argument('--verbose', action='store_false')
    
    args = parser.parse_args()
    
    # map_selected_models = text_based_matching(args.metadata_path, args.symbol_description_json_file, 
    #                                           use_shape=True, use_keywords=True, use_long_description=False)
    
    # if args.verbose:    
    #     for map_name, selected_models in map_selected_models.items():    
    #         print(map_name, map_selected_models[map_name])