import json
import os 
import numpy as np


def remove_unicode(text):
    # Create an empty string to store the cleaned text
    cleaned_text = ''
    
    for char in text:
        if char < 128:  # Check if the character is ASCII
            cleaned_text += chr(char)
        elif len(cleaned_text) > 0 and cleaned_text[-1] != ' ':
            cleaned_text += ' '
    return cleaned_text

def generate_roi(map_name, json_dir='/data/weiweidu/criticalmaas_data/training', buf=1800):
    regions = []
    json_path = os.path.join(json_dir, map_name+'.json')
    f = open(json_path)
    metadata = json.load(f)
    for symbol in metadata['shapes']:
        if '_line' in symbol['label'].lower():
            regions.append(symbol['points'][0])
            regions.append(symbol['points'][1])
    regions = np.array(regions).astype('int32')
    # x, y are swapped
    xmin, xmax = np.min(regions[:,0]), np.max(regions[:,0])
    ymin, ymax = np.min(regions[:,1]), np.max(regions[:,1])
    return [xmin-300, ymin-300, xmax+buf, ymax+300]

def non_max_suppression(bboxes, confidence_threshold=0.5, iou_threshold=0.2):
    # Sort bounding boxes by confidence in descending order
#     bboxes.sort(key=lambda x: x[4], reverse=True)

    selected_bboxes = []

    while len(bboxes) > 0:
        current_box = bboxes[0]
        selected_bboxes.append(current_box)
        bboxes = bboxes[1:]

        for box in bboxes:
            iou = calculate_iou(current_box, box)
            if iou > iou_threshold:
                xmin, ymin = min(box[0], current_box[0]), min(box[1], current_box[1])
                xmax, ymax = max(box[2], current_box[2]), max(box[3], current_box[3])
                selected_bboxes[-1] = [xmin, ymin, xmax, ymax]
                bboxes.remove(box)

    return selected_bboxes

def calculate_iou(box1, box2):
    # Calculate the Intersection over Union (IoU) between two bounding boxes
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    w1, h1 = x12 - x11, y12 - y11
    w2, h2 = x22 - x21, y22 - y21

    x_intersection = max(0, min(x11 + w1, x21 + w2) - max(x11, x21))
    y_intersection = max(0, min(y11 + h1, y21 + h2) - max(y11, y21))

    area_intersection = x_intersection * y_intersection
    area_union = w1 * h1 + w2 * h2 - area_intersection

    iou = area_intersection / area_union
    return iou
