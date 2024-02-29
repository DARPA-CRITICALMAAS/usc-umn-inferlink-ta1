import pytesseract
from PIL import Image
import json
import os
import numpy as np
from utils import ocr_bbox
from difflib import SequenceMatcher

Image.MAX_IMAGE_PIXELS = 933120000

def str_similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_symbol_names(map_image, all_symbol_bbox, width):
    img_h, img_w = map_image.shape[:2]
    bbox_ocr = {}
    if all_symbol_bbox == {}:
        return bbox_ocr
    all_symbol_bbox.sort(key=lambda x: x[1])

    for i, bbox in enumerate(all_symbol_bbox):
        x1, y1, x2, y2 = bbox
        if i < len(all_symbol_bbox) -1:
            next_row = all_symbol_bbox[i+1][1]
        else:
            next_row = y2+100
        img4ocr = map_image[y1:min(y2+100, next_row, img_h), x2:x2+min(2000,width, img_w)]
        ocr_res = ocr_bbox(img4ocr)
        bbox_ocr[str(bbox)] = ocr_res
    return bbox_ocr

def match_str(query_words, source_str):
    num_matched = 0
    for word in query_words:
        if word in source_str:
            num_matched += 1.0
    return num_matched/len(query_words)

def _match_symbol(ocr_res, gpt_res, matched_dict):
    matched_score = 0
    
    query_words = ocr_res.split(' ')
    
    matched_symbol_name = None
    matched_description = None
    
    for sym, des in gpt_res.items():
        if len(des) > 500:
            score1 = match_str(query_words, des[:len(des)//2])
        else:
            score1 = match_str(query_words, des)
        score2 = match_str(query_words, sym)
        score = score1 + score2
        if score > matched_score:# and not matched_dict[sym]: 
            matched_score = score
            matched_symbol_name = sym
            matched_description = des
        
    return matched_symbol_name, matched_description

def match_symbol(ocr_res, gpt_res, matched_dict=None):
    matched_score = 0
    matched_symbol_name = None
    matched_description = None
    
    if not gpt_res:
        return matched_symbol_name, matched_description 
    
    for sym, des in gpt_res.items():
        des_match_score = str_similar(ocr_res, des) if len(des) != 0 else 0
        sym_match_score = str_similar(ocr_res, sym)  if len(sym) != 0 else 0          
        sim_score = max(des_match_score, sym_match_score)
        
        if sim_score > matched_score:
            matched_score = sim_score
            matched_symbol_name = sym
            matched_description = des
    if matched_score < 0.5:
        return None, None
    else:
        return matched_symbol_name, matched_description    

def match_orc_gpt_results(bbox_ocr_dict, gpt_res, map_image):
#     matched_flag = {k: False for k in gpt_res}
    matched_res = {}
    for bbox, ocr_res in bbox_ocr_dict.items():
        x1, y1, x2, y2 = eval(bbox)
        bbox_img = map_image[y1:y2, x1:x2]

        if len(ocr_res) == 0 or np.var(bbox_img) < 100:
            continue
#         matched_symbol_in_gpt, matched_desc_in_gpt = match_symbol(ocr_res, gpt_res, matched_flag)
        matched_symbol_in_gpt, matched_desc_in_gpt = match_symbol(ocr_res, gpt_res)
#         matched_flag[matched_symbol_in_gpt] = True
#         print(f'*** {matched_desc_in_gpt} *** ')
#         print(ocr_res)
#         print('=======')
        if matched_symbol_in_gpt:
            matched_res[str(bbox)] = {'description': gpt_res[matched_symbol_in_gpt], 'symbol name': matched_symbol_in_gpt}
        else:
            matched_res[str(bbox)] = {'description': ocr_res, 'symbol name': ocr_res}
    return matched_res

if __name__ == '__main__':
    map_legend_dir = '/data/weiweidu/gpt4/map_legend_gpt_input'
    map_legend_name = 'MA_Grafton_9989_3582_1597_1747'
    map_dir = '/data/weiweidu/criticalmaas_data/validation'
    
#     symbol_names = get_symbol_names_within_roi(map_legend_dir, map_legend_name, map_dir)
#     print(symbol_names)
    symbol_texts, bboxes = get_symbol_names_within_roi_with_buffer(map_legend_dir, map_legend_name, map_dir)

    