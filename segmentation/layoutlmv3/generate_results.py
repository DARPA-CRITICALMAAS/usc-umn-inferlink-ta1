import os
import json
from PIL import Image
import numpy as np
import pytesseract
import NMS

def read_prediction_json(json_path):
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict

def read_annotation_json(json_path):
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)
    image_dict = {}
    for item in json_dict['images']:
        image_dict[item['id']] = item['file_name']
    return image_dict

def ocr_on_bbox(bbox_img):   
    ocr = pytesseract.image_to_data(bbox_img, output_type=pytesseract.Output.DICT)
#     print(ocr['text'])
    res = ""
    for i in range(len(ocr['text'])):
        if int(ocr['conf'][i]) > 0.1 and ocr['text'][i] != '' and not ocr['text'][i].isspace():  # Confidence threshold.
            text = ocr['text'][i]#.encode('ascii', 'ignore').decode('ascii')
            res += ' ' + text
    return res

def post_process_nms(image_dir, img_dict, pred_json_path, refined_output_path):
    pred_dict = read_prediction_json(pred_json_path)
    refined_layoutlmv3_output = []
    for target_image_id, image_name in img_dict.items():
        image_path = os.path.join(image_dir, image_name)
        pred_symb_bboxes, pred_symb_scores = [], []
        pred_desc_bboxes, pred_desc_scores = [], []
        pred_legd_bboxes, pred_legd_scores = [], []
        
        for item in pred_dict:
            image_id = item['image_id']   
            if item['score'] > 0.1 and image_id == target_image_id:
                x, y, w, h = np.array(item['bbox']).astype('int16')
                x1, y1, x2, y2 = x, y, x+w, y+h
                if item['category_id'] == 3:
                    pred_legd_bboxes.append([x1, y1, x2, y2])
                    pred_legd_scores.append(item['score'])
                elif item['category_id'] == 2:
                    pred_desc_bboxes.append([x1, y1, x2, y2])
                    pred_desc_scores.append(item['score'])
                elif item['category_id'] == 1:
                    pred_symb_bboxes.append([x1, y1, x2, y2])
                    pred_symb_scores.append(item['score'])

        refined_symb_bboxes_indices = NMS.non_max_suppression(pred_symb_bboxes, pred_symb_scores, 0.3)
        refined_desc_bboxes_indices = NMS.non_max_suppression(pred_desc_bboxes, pred_desc_scores, 0.3)
        refined_legd_bboxes_indices = NMS.non_max_suppression(pred_legd_bboxes, pred_legd_scores, 0.3)

        for i in refined_symb_bboxes_indices:
            temp_dict = {
                'image_id': target_image_id,
                'category_id': 1,
                'bbox': [int(pred_symb_bboxes[i][0]), int(pred_symb_bboxes[i][1]), \
                         int(pred_symb_bboxes[i][2]), int(pred_symb_bboxes[i][3])],
                'score': pred_symb_scores[i]
            }
            refined_layoutlmv3_output.append(temp_dict)
        for i in refined_desc_bboxes_indices:   
            temp_dict = {
                'image_id': target_image_id,
                'category_id': 2,
                'bbox':[int(pred_desc_bboxes[i][0]), int(pred_desc_bboxes[i][1]), \
                        int(pred_desc_bboxes[i][2]), int(pred_desc_bboxes[i][3])],
                'score': pred_desc_scores[i]
            }
            refined_layoutlmv3_output.append(temp_dict)
        for i in refined_legd_bboxes_indices:
            temp_dict = {
                'image_id': target_image_id,
                'category_id': 3,
                'bbox': [int(pred_legd_bboxes[i][0]), int(pred_legd_bboxes[i][1]), \
                        int(pred_legd_bboxes[i][2]), int(pred_legd_bboxes[i][3])],
                'score': pred_legd_scores[i]
            }
            refined_layoutlmv3_output.append(temp_dict)
    with open(refined_output_path, 'w') as f:
        json.dump(refined_layoutlmv3_output, f)
    return refined_output_path
        
def process_layoutlmv3_outputs(pred_json_path, image_dir, annotation_json_path, tif_map_name, output_dir):
    img_dict = read_annotation_json(annotation_json_path)
    refined_output_path = pred_json_path[:-5] + '_refined.json'
    post_process_nms(image_dir, img_dict, pred_json_path, refined_output_path)
    pred_dict = read_prediction_json(refined_output_path)

    output_path_point = os.path.join(output_dir, tif_map_name+'_point.json')
    output_path_line = os.path.join(output_dir, tif_map_name+'_line.json')
    output_dict = {}
    for target_image_id, image_name in img_dict.items():
        image_path = os.path.join(image_dir, image_name)
        image_pil = Image.open(image_path)
        image = np.array(image_pil)   
        for item in pred_dict:
            image_id = item['image_id']   
            if item['score'] > 0.1 and image_id == target_image_id:
                x, y, w, h = np.array(item['bbox']).astype('int16')
                if w == 0 or h == 0:
                    continue
                bbox_img = image[y:y+h, x:x+w]
                text = ocr_on_bbox(bbox_img)
                output_dict[str([x,y,w,h])] = text
    with open(output_path_point, 'w') as f:
        json.dump(output_dict, f)
    with open(output_path_line, 'w') as f:
        json.dump(output_dict, f)      
    return 0
        
if __name__ == "__main__":
    pred_json_path = '/ta1/dev/unilm/layoutlmv3_inputs_outputs/inference/coco_instances_results.json'
    image_dir = '/ta1/dev/unilm/layoutlmv3_inputs_outputs/validation'
    annotation_json_path = '/ta1/dev/unilm/layoutlmv3_inputs_outputs/validation.json'
    output_dir = '/ta1/dev/unilm/layoutlmv3_inputs_outputs/system_outputs_validation'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    process_layoutlmv3_outputs(pred_json_path, image_dir, annotation_json_path, output_dir)
    

