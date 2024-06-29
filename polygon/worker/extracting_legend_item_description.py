import os
from legend_item_description_extractor.gpt4_input_generation import main as generate_gpt4_input
from legend_item_description_extractor.symbol_description_extraction_gpt4 import main as extract_symbol_description
from legend_item_description_extractor.postprocess import combine_json_files_from_gpt


def extracting_legend_item_description(dir_to_map, map_name, dir_to_json, dir_to_roi, dir_to_gpt4, dir_to_gpt4_intermediate):
    all_sym_bbox = generate_gpt4_input(dir_to_map, dir_to_roi, dir_to_json, map_name, dir_to_gpt4)

    for root, dirs, files in os.walk(dir_to_gpt4):
        for f_name in files:
            if map_name not in f_name:
                continue
            image_name = f_name.split('.')[0]
            extract_symbol_description(image_name, dir_to_map, all_sym_bbox, dir_to_gpt4, dir_to_gpt4_intermediate)
    
    for item_category in ['polygon']:
        output_path = os.path.join(dir_to_gpt4_intermediate, map_name+'_'+str(item_category)+'.json')
        combine_json_files_from_gpt(dir_to_gpt4_intermediate, map_name, output_path, item_category)
    
    return True, map_name
