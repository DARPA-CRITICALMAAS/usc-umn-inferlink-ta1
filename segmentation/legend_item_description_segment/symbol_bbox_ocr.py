import pytesseract
from PIL import Image
import json
import os
import numpy as np

Image.MAX_IMAGE_PIXELS = 933120000

def get_bbox_within_roi(all_boxes, roi_bbox):
    bboxes_in_roi = []
    x, y, w, h = roi_bbox
    
    for x1, y1, x2, y2 in all_boxes:
        if x1 >= x and y1 >= y and x2 <= x+w and y2 <= y+h:
            bboxes_in_roi.append([x1,y1,x2,y2])
    return bboxes_in_roi


def get_symbol_ocr(map_path, bbox):
    map_image_pil = Image.open(map_path)
    map_image = np.array(map_image_pil)
    h, w = map_image.shape[:2]
    x1, y1, x2, y2 = bbox
    symbol_image = map_image[y1:min(y2, h), x1:min(x2, w)]
#     print(y1, min(y2, h), x1, min(x2, w), h, w)
    symbol_image_pil = Image.fromarray(np.uint8(symbol_image))
#     symbol_image_pil.save(f'temp/{x1}_{y1}_{x2}_{y2}.png')
    
    ocr = pytesseract.image_to_data(symbol_image, output_type=pytesseract.Output.DICT)
#     print(ocr['text'])
    res = ""
    for i in range(len(ocr['text'])):
        if int(ocr['conf'][i]) > 0.1 and ocr['text'][i] != '' and not ocr['text'][i].isspace():  # Confidence threshold.
            text = ocr['text'][i]#.encode('ascii', 'ignore').decode('ascii')
            res += ' ' + text
#             if not text.isspace() or text == '':
#                 return text
    return res
    
def get_symbol_names_within_roi(map_legend_dir, map_legend_name, map_dir):
    temp_legend_names = map_legend_name.split('_')
    x, y, w, h = int(temp_legend_names[-4]), int(temp_legend_names[-3]), int(temp_legend_names[-2]), int(temp_legend_names[-1])
    map_name = '_'.join(i for i in temp_legend_names[:-4])
    
    json_path = os.path.join(map_dir, map_name+'.json')
    map_path = os.path.join(map_dir, map_name+'.tif')
    
    symbol_bbox_list = get_bbox_within_roi(json_path, [x, y, w, h])
    
    symbol_name_list = []
    bbox_list = []
    for x1,y1,x2,y2 in symbol_bbox_list:
        symbol_name = get_symbol_ocr(map_path, [x1,y1,x2,y2])
        symbol_name_list.append(symbol_name)
        bbox_list.append([x1,y1,x2,y2])
#         if symbol_name is not None:
#             
    return symbol_name_list, bbox_list

def get_symbol_names_within_roi_with_buffer(map_legend_dir, map_legend_name, map_dir, all_boxes_dict):
    temp_legend_names = map_legend_name.split('_')
    x, y, w, h = int(temp_legend_names[-4]), int(temp_legend_names[-3]), int(temp_legend_names[-2]), int(temp_legend_names[-1])
    map_name = '_'.join(i for i in temp_legend_names[:-4])
    
    map_path = os.path.join(map_dir, map_name+'.tif')
    
    symbol_text_dict = {'line': [], 'point': [], 'polygon': []}
    bbox_dict = {'line': [], 'point': [], 'polygon': []}
    
    for category in ['line', 'point', 'polygon']:
        all_boxes = all_boxes_dict[category]
        symbol_bbox_list = get_bbox_within_roi(all_boxes, [x, y, w, h])
        symbol_text_dict[category] = symbol_bbox_list
    
        symbol_text_list, bbox_list = [], []
        print(f'# bboxes = {len(symbol_bbox_list)}')
        for x1, y1, x2, y2 in symbol_bbox_list:
            symbol_text = get_symbol_ocr(map_path, [x2, y1, x2+w, y2])
            print(symbol_text)
    #         add_factor, count = 30, 1
    #         while symbol_text.isspace():
    #             add_buf = add_factor * count
    #             symbol_text = get_symbol_ocr(map_path, [x1, y1, x2+200+add_buf*count, y2+10])
    #             count += 1
            print('=========')
            symbol_text_list.append(symbol_text)
            bbox_list.append([x1,y1,x2,y2])
        bbox_dict[category] = bbox_list
        symbol_text_dict[category] = symbol_text_list
    return symbol_text_dict, bbox_dict

def get_symbol_names_within_roi_with_buffer_v2(map_legend_dir, map_legend_name, map_dir, all_boxes):
    '''
    v2 considers the positions of 
    other bboxes, the buffer does not
    cover other bboxes
    '''
    temp_legend_names = map_legend_name.split('_')
    x, y, w, h = int(temp_legend_names[-4]), int(temp_legend_names[-3]), int(temp_legend_names[-2]), int(temp_legend_names[-1])
    map_name = '_'.join(i for i in temp_legend_names[:-4])
    
    map_path = os.path.join(map_dir, map_name+'.tif')
    
    _symbol_bbox_list = get_bbox_within_roi(all_boxes, [x, y, w, h])
    _symbol_bbox_list.sort(key=lambda x:x[1])
    
    # remove duplicates
    symbol_bbox_list = []
    for i in _symbol_bbox_list:
        if i not in symbol_bbox_list:
            symbol_bbox_list.append(i)
    
    symbol_text_list = []
    bbox_list = []
    print(f'# bboxes = {len(symbol_bbox_list)}')
    for i, bbox in enumerate(symbol_bbox_list):
        x1, y1, x2, y2 = bbox
        
        if i + 1 < len(symbol_bbox_list):
            next_y1 = symbol_bbox_list[i+1][1]
#             print('bbox1: ', x1, y1, x2, y2, x1, y1, x1+min(w,1000), max(next_y1, y2))
#             symbol_text = get_symbol_ocr(map_path, [x1, y1, x1+min(w,1000), next_y1])
            symbol_text = get_symbol_ocr(map_path, [x2, y1, x2+min(w,1000), next_y1])
        else:
#             print('bbox2: ', x1, y1, x2, y2, x1, y1, x1+w, y+h)
            symbol_text = get_symbol_ocr(map_path, [x2, y1, x2+w, y+h])
        print(symbol_text)    
        print('=========')
        symbol_text_list.append(symbol_text)
        bbox_list.append([x1,y1,x2,y2])
    return symbol_text_list, bbox_list

def get_symbol_names(map_legend_dir, map_legend_name, map_dir, all_symbol_bbox):
    '''
    A wrap-up for 
    func get_symbol_names_within_roi_with_buffer_v2()
    & funcget_symbol_names_within_roi_with_buffer()
    if #symbols > 30:
        use get_symbol_names_within_roi_with_buffer()
    else:
        use get_symbol_names_within_roi_with_buffer_v2()
    
    '''
    temp_legend_names = map_legend_name.split('_')
    x, y, w, h = int(temp_legend_names[-4]), int(temp_legend_names[-3]), int(temp_legend_names[-2]), int(temp_legend_names[-1])
    map_name = '_'.join(i for i in temp_legend_names[:-4])
    
    map_path = os.path.join(map_dir, map_name+'.tif')

    symbol_text_list, bbox_list = get_symbol_names_within_roi_with_buffer(map_legend_dir, \
                                                                              map_legend_name, map_dir, all_symbol_bbox)
    return symbol_text_list, bbox_list

if __name__ == '__main__':
    map_legend_dir = '/data/weiweidu/gpt4/map_legend_gpt_input'
    map_legend_name = 'MA_Grafton_9989_3582_1597_1747'
    map_dir = '/data/weiweidu/criticalmaas_data/validation'
    
#     symbol_names = get_symbol_names_within_roi(map_legend_dir, map_legend_name, map_dir)
#     print(symbol_names)
    symbol_texts, bboxes = get_symbol_names_within_roi_with_buffer(map_legend_dir, map_legend_name, map_dir)

    