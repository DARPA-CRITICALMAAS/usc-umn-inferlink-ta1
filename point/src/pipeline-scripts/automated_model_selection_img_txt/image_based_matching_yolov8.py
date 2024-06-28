### 
# This script is mainly working for high pirority symbols
###

import os
import json
import shutil
import glob
import argparse
from PIL import Image
from ultralytics import YOLO


classes = ["drill_hole", 
           "gravel_pit", 
           "prospect", 
           "quarry", 
           "mine_shaft", 
           "mine_tunnel",
           "inclined_bedding",
           "vertical_bedding",
           "horizontal_bedding",
           "overturned_bedding",
           "inclined_metamorphic",
           "vertical_metamorphic",
           "inclined_metamorphic_with_lineation",
           "inclined_flow_banding",
           "vertical_flow_banding",
           "lineation"]

target_point_symbols = [
           "drill_hole", 
           "gravel_pit", 
           "prospect", 
           "quarry", 
           "mine_shaft", 
           "mine_tunnel",
           "inclined_bedding",
           "overturned_bedding",
           "inclined_metamorphic",
           "inclined_flow_banding",
           "lineation"]


def image_based_matching(image_path, weight_path, conf=0.4):
    print(image_path,'crop image path')
    images = glob.glob(os.path.join(image_path, '*.png'))
    if len(images) == 0: return []
    print(weight_path)
    model = YOLO(weight_path)
    selected_models = []
    for each in images:
        results = model.predict(each, iou=0., conf=conf, stream=True, verbose=False)    
        for i, r in enumerate(results):
            boxes = r.boxes.cpu()
            boxes2 = []
            for box in boxes:
                x1, y1, x3, y3 = box.xyxy.numpy().reshape(-1).astype(int)
                if x1 - 5 < 0 or y1 - 5 < 0: continue;
                if x3 + 5 > 640 or y3 + 5 > 640: continue;
                if y3 - y1 > 100: continue;
                boxes2.append(box)
            boxes = boxes2
            
            for box in boxes:
                c = classes[int(box.cls.numpy()[0])]
                if c in target_point_symbols:
                    if c not in selected_models:
                        selected_models.append(c)
                        # print(selected_models)
    
    return list(set(selected_models))
    
    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='symbol')
#     parser.add_argument('--image_path', type=str, default="")
#     parser.add_argument('--weight_path', type=str, default="./model_selection.pt")
#     parser.add_argument('--conf', type=float, default=0.4)
#     args = parser.parse_args()
    
#     all_maps = glob.glob("/home/yaoyi/shared/critical-maas/ta1_feature_extraction/evaluation/*.tif")
#     all_maps = sorted([os.path.basename(m) for m in all_maps])

#     root = '/home/yaoyi/lin00786/data/critical-mass/legend_items/eval_hackathon/images'
    
#     out = {}
#     for map_filename in all_maps:
#         map_name = map_filename.replace('.tif', '')        
#         args.image_path = os.path.join(root, map_name)
#         map_selected_models = image_based_matching(args.image_path, args.weight_path, args.conf)
#         out[map_name] = map_selected_models
        
#     with open(f'../image_based_selection.json', 'w') as f:
#         json.dump(out, f)
    