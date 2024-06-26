import os
import json
from PIL import Image
import numpy as np
import pytesseract

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

def main(pred_json_path, image_dir, annotation_json_path, output_dir):
    pred_dict = read_prediction_json(pred_json_path)
    img_dict = read_annotation_json(annotation_json_path)
    for target_image_id, image_name in img_dict.items():
        image_path = os.path.join(image_dir, image_name)
        image_pil = Image.open(image_path)
        image = np.array(image_pil)
        map_name = "_".join(image_name.split('_')[:-4])
        output_path_point = os.path.join(output_dir, map_name+'point.json')
        output_path_line = os.path.join(output_dir, map_name+'line.json')
        output_dict = {}
        for item in pred_dict:
            image_id = item['image_id']   
            if item['score'] > 0.1 and image_id == target_image_id:
                x, y, w, h = np.array(item['bbox']).astype('int16')
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
    main(pred_json_path, image_dir, annotation_json_path, output_dir)
    

