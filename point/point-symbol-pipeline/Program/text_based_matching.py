# import editdistance
import os
import json
 import argparse  
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 

def text_based_matching(text_desc_json_dir_path,input_map_dir_path):
    #keywords per each symbol (following pretrained model names)

    keywords_per_symbol={"quarry":['open','pit','mine','quarry'],"gravel_pit":['gravel','sand','clay','borrow','pit'],"mine_tunnel":['tunnel','mine','cave','entrance','adit'],
    "mine_shaft":['mine','shaft'],"prospect":['prospect'],
    "inclined_metamorphic":['inclined','metamorphic','tectonic','foliation','strike','dip','incline'],
    "drill_hole":["drill","hole","mineral","exploration"],"inclined_flow_banding":["inclined","flow","banding",'strike','dip','incline'],
    "lineation":["lineation","plunge","arrow","direction","linear","linear structure"],"inclined_bedding":["inclined","bedding","stike","dip","bed"],"overturned_bedding":['overturned','bedding','strike','dip',"bed"]}

    json_dir=text_desc_json_dir_path
    map_patches_list=os.listdir(input_map_dir_path)
    res_dict={}
    for each_map in map_patches_list:  
        for each_json in os.listdir(json_dir):
            if each_json.startswith(each_map):
                if each_map not in res_dict.keys():
                    res_dict[each_map]={}
                f=open(os.path.join(json_dir,each_json))
                dict_per_map=json.load(f)
                for k,v in dict_per_map.items():
                    if type(v)==dict:
                        k_list=v["symbol name"].split(' ')
                        v_list=v["description"].split(' ')
                        k_list=k_list+v_list
                        stop_words = stopwords.words('english') 
                        cleaned_k_list = []
                        for word in k_list:
                            if word not in stop_words:
                                cleaned_k_list.append(word)
                        k_list=cleaned_k_list
                        for model_name in keywords_per_symbol.keys():
                            tmp_num=0
                            for each_char in k_list:  
                                each_char=each_char.lower()                        
                                each_char=each_char.replace(',','')
                                each_char=each_char.replace('.','') 
                                for each_key in keywords_per_symbol[model_name]:  
                                    if each_char == each_key:
                                        tmp_num+=1
                                    ##will be improved 
                                    # elif editdistance.eval(each_char, each_key)<=1:
                                    #     tmp_num+=1 
                            if model_name in res_dict[each_map].keys():
                                res_dict[each_map][model_name]+=tmp_num
                            else:
                                res_dict[each_map][model_name]=tmp_num
    matched_model={}
    for key in res_dict.keys():
        matched_model[key]=[]
    for each_map,pnt_dict in res_dict.items():
        for pt_model,each_cnt in pnt_dict.items():
            if each_cnt>0:
                matched_model[each_map].append(pt_model+'.pt')

    return matched_model

