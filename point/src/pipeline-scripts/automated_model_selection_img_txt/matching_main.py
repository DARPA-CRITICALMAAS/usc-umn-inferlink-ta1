import json
import os
import argparse
import glob

from .text_based_matching_tfidf import text_based_matching
from .image_based_matching_yolov8 import image_based_matching

def txt_img_main(map_name,
         use_image_based,
         use_text_based,
         symbol_description_json_file,
         metadata_path,
         metadata_type,
         cropped_image_root,
         image_based_model_weight,
         image_based_model_conf):

    if cropped_image_root is None :
        print("Cropped image path does not exist.")
        use_image_based = False 

    for each_map_crop_dir in os.listdir(cropped_image_root):
        if map_name in each_map_crop_dir:
            cropped_image_path =  os.path.join(cropped_image_root,each_map_crop_dir)
    # if not os.path.exists(cropped_image_path):
    #     print(cropped_image_path)
    #     os.mkdir(cropped_image_path)

    if metadata_path is None or not os.path.exists(metadata_path):
        print("Metadata does not exist.")
        use_text_based = False        
    # if cropped_image_path is None or not os.path.exists(cropped_image_path):
    #     print("Cropped image path does not exist.")
    #     use_image_based = False          


    text_based_selection, image_based_selection = [], []
    if use_text_based:
        text_based_selection = text_based_matching(map_name, 
                                                    metadata_path, 
                                                    metadata_type, 
                                                    symbol_description_json_file,
                                                    use_shape=metadata_type=='gpt', 
                                                    use_keywords=True, 
                                                    use_long_description=False)
        print('Text_based:', text_based_selection)
        
    if use_image_based:
        image_based_selection = image_based_matching(cropped_image_path, 
                                                      image_based_model_weight, 
                                                      image_based_model_conf)
        print('Image_based:', image_based_selection)
        
    selected_models = list(set(text_based_selection + image_based_selection))
    for idx in range(len(selected_models)):
        selected_models[idx] = selected_models[idx] +'.pt'

    return selected_models
    
    

if __name__ == "__main__":
    #cmd 
    # python matching_main.py --metadata_path --cropped_image_path --map_name --metadata_type [gpt/layout] --image_based_model_weight --symbol_description_json_file 

    # basic parameters
    parser = argparse.ArgumentParser(description='symbol')
    parser.add_argument('--map_name', type=str, default="")
    parser.add_argument('--use_image_based', action='store_false')
    parser.add_argument('--use_text_based', action='store_false')
    parser.add_argument('--symbol_description_json_file', type=str, default="/home/leeje/critical-maas/automated-matching/automated-module/automated_model_selection_img_txt/data/symbol_info.json")
    parser.add_argument('--metadata_path', type=str, default="../point_metadata_nickel")
    parser.add_argument('--metadata_type', type=str, default="gpt")
    parser.add_argument('--cropped_image_path', type=str, default="")
    parser.add_argument('--image_based_model_weight', type=str, default="/home/leeje/critical-maas/automated-matching/automated-module/automated_model_selection_img_txt/data/model.pt")
    parser.add_argument('--image_based_model_conf', type=float, default=0.4)
    
    args = parser.parse_args()
    
    out = {}
    map_name = args.map_name   
    cropped_image_path = args.cropped_image_path

    map_selected_models = txt_img_main(map_name,
                                args.use_image_based,
                                args.use_text_based,
                                args.symbol_description_json_file,
                                args.metadata_path,
                                args.metadata_type,
                                cropped_image_path,
                                args.image_based_model_weight,
                                args.image_based_model_conf)

    out[map_name] = map_selected_models
    print("Final output:", map_name, map_selected_models)
    print('\n')
        
