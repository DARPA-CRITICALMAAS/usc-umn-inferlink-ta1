import os
from postprocessing.remove_pnts_from_text_legend import *
from automated_model_selection.text_based_matching_tfidf import *
from prediction_stitch.pred_stitch_point import * 
import time
import pickle
import os
import time
import argparse
import logging


def parse_arguments():
    parser = argparse.ArgumentParser(description="Point Feature Pipeline Path.")

    parser.add_argument("--crop_shift_size", type=int, default=1000,
                        help="shift size for cropped image patches")
    parser.add_argument("--map_dir", type=str, default="",
                        help="Directory to the original map data ended with .tif.")
    parser.add_argument("--map_metadata_dir", type=str, default="",
                        help="Directory to map metadata that contains the information about coordinate of legend and map dimension.")
    parser.add_argument("--cropped_legend_dir", type=str, default="",
                        help="Directory to save the cropped legend per each map.")
    parser.add_argument("--template_dir", type=str, default="template/",
                        help="Directory to the legend template.")
    parser.add_argument("--processed_legend_dir", type=str, default="",
                        help="Directory to save the processed cropped legend")
    parser.add_argument("--map_patches_dir", type=str, default="",
                        help="Root directory for input map patches directory.")
    parser.add_argument("--model_weights_dir", type=str, default="model_weights/",
                        help="Directory to pretrained point feature detection model weights.")
    parser.add_argument("--text_spotting_dir", type=str, default="",
                        help="mapKurator output dir containing text spotting results.")
    parser.add_argument("--symbol_info_json_file", type=str, default="automated_model_selection/symbol_info.json",
                        help="Json file contains internal metadata for point symbols.")
    parser.add_argument("--output_dir_root", type=str, default="",
                        help="Root directory for output directory.")

    return parser.parse_args()

logger = logging.getLogger(__name__)
FileOutputHandler = logging.FileHandler('logs_point.txt')
logger.addHandler(FileOutputHandler)

start_time = time.time()

args = parse_arguments()

metadata_path = args.map_metadata_dir
spotter_path = args.text_spotting_dir
input_dir_root = args.map_patches_dir
model_weights_dir = args.model_weights_dir
output_dir_root = args.output_dir_root
symbol_info_json_file = args.symbol_info_json_file
crop_shift_size = args.crop_shift_size


predict_output_dir = os.path.join(output_dir_root, 'prediction')
if not os.path.exists(predict_output_dir):
    os.makedirs(predict_output_dir)

stitch_output_dir = os.path.join(output_dir_root, 'stitch-per-symbol')
if not os.path.isdir(stitch_output_dir):
    os.mkdir(stitch_output_dir)

final_output_dir = os.path.join(output_dir_root, 'output-per-symbol')
if not os.path.isdir(final_output_dir):
    os.mkdir(final_output_dir)

input_map_name = os.path.basename(os.path.dirname(input_dir_root+'/'))
input_map_name =input_map_name.replace('_g1000_s1000','')
print(input_map_name)

map_selected_models =[]
try:
    map_selected_models = text_based_matching(input_map_name, metadata_path, symbol_info_json_file, 
                                        use_shape=True, use_keywords=True, use_long_description=False)
    print(map_selected_models)
except Exception as Argument:
    logger.warning("Problems in pretrained models selection module")

# for mapname, point_symbol in model_dict.items():
selected_model_weights = map_selected_models 
map_input_dir_root=input_dir_root
if os.path.exists(map_input_dir_root):
    try: 
        print("=== Running a model prediction module ===")
        predict_img_patches(input_map_name, map_input_dir_root, model_weights_dir, selected_model_weights, predict_output_dir)
    except Exception as Argument:
        logger.warning("Problems in point symbol prediction module: {0}".format(map_input_dir_root))
    try:
        print("=== Running a stitching module ===")
        stitch_to_each_point(input_map_name, map_input_dir_root, predict_output_dir, stitch_output_dir, crop_shift_size)
    except Exception as Argument:
        logger.warning("Problems in point symbol stitching module: {0}".format(map_input_dir_root))
else:
    logger.warning("No cropped image patches exists : {0}".format(map_input_dir_root))    

print(" === Running a postprocessing module === ")
try: 
    stitch_output_dir_per_map=os.path.join(stitch_output_dir,input_map_name)
    postprocessing(stitch_output_dir_per_map, metadata_path, spotter_path, final_output_dir, if_filter_by_text_regions=False)
except Exception as Argument:
    logger.warning("Problems in postprocessing module :{0}".format(each_map))  

print(" === Done processing point symbol pipeline === ")
end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")

## {map_name}_point.json.
##--cropped_legend_dir /your/legend/directory --template_dir /your/template/directory --processed_legend_dir /your/processed/legend/directory
# python run_final.py --map_dir /your/map/directory --map_metadata_dir /your/metadata/directory  --map_patches_dir /your/patches/directory --model_weights_dir /your/weights/directory --output_dir_root /your/output/root/directory
# python run_point_pipe.py --map_metadata_dir /home/yaoyi/jang0124/critical-maas/point-symbol-pipeline/main-pipeline/sample_run/metadata/ --map_patches_dir /home/yaoyi/jang0124/critical-maas/point-symbol-pipeline/main-pipeline/sample_run/map_patches/169_34067_g1000_s1000/ --model_weights_dir /home/yaoyi/jang0124/critical-maas/code/pipeline/model-name-official/Model_weights/ --output_dir_root /home/yaoyi/jang0124/critical-maas/point-symbol-pipeline/main-pipeline/sample_run/out-per-map/ 
