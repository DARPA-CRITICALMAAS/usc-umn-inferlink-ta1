from generate_Labels_new import *
from similarity_match import *
from Seperate_symbol_text import *
from collections import defaultdict
from pred_stitch_point_updated import *
from geojson_to_raster import *
from text_based_matching import *
from post_process_geojson import *
import os
import time
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Point Feature Pipeline Path.")

    parser.add_argument("--map_dir", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/data/hackthon2/raw_map",
                        help="Directory to the original map data ended with .tif.")
    parser.add_argument("--map_metadata_dir", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/data/hackthon2/meta_data",
                        help="Directory to map metadata that contains the information about coordinate of legend and map dimension.")
    parser.add_argument("--cropped_legend_dir", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/data/hackthon2/vallabels",
                        help="Directory to save the cropped legend.")
    parser.add_argument("--template_dir", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/template_priority",
                        help="Directory to the legend template.")
    parser.add_argument("--processed_legend_dir", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/data/hackthon2/vallabels_processed",
                        help="Directory to save the processed cropped legend")
    parser.add_argument("--map_patches_dir", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/data/hackthon2/hackathon2_cropped",
                        help="Root directory for input map patches directory.")
    parser.add_argument("--model_weights_dir", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/model_weights_12_240118",
                        help="Directory to pretrained point feature detection model weights.")
    parser.add_argument("--output_dir_root", type=str, default="/Users/dong_dong_dong/Downloads/Darpa/data/hackthon2/model_output_pipeline",
                        help="Root directory for output directory.")

    return parser.parse_args()

start_time = time.time()

args = parse_arguments()

raw_map_path = args.map_dir
metadata_path = args.map_metadata_dir
cropped_path = args.cropped_legend_dir
template_dir = args.template_dir
processed_path = args.processed_legend_dir
input_dir_root = args.map_patches_dir
model_weights_dir = args.model_weights_dir
output_dir_root = args.output_dir_root

# crop_legned(raw_map_path, metadata_path, cropped_path)

# target_color = [255, 255, 255]
# clean_background_and_remove_text(cropped_path, processed_path, target_color, threshold=50, crop=False, use_gpu=False)

# index_list_dict_rank = text_based_matching(metadata_path,raw_map_path)
index_list_dict_rank = text_based_matching(metadata_path,input_dir_root)
model_dict=index_list_dict_rank

# index_list_dict_rank = image_similarity(template_dir, processed_path, rank = 1)

# files = glob.glob(os.path.join(processed_path, '*.jpeg'))
# # Extract filenames without the path
# filenames = [os.path.basename(file).split('_label_')[0] for file in files]

# # Sort the filenames
# sorted_filenames = sorted(filenames)
# model_dict = defaultdict(set)

# for i, map in enumerate(sorted_filenames):
#     found_keys = [key + ".pt" for key, val_list in index_list_dict_rank.items() if i in val_list]
#     model_dict[map].update(found_keys)

# model_dict = {key: {value + ".pt" for value in values} for key, values in model_dict.items()}
print(model_dict)

predict_output_dir = os.path.join(output_dir_root, 'prediction')
if not os.path.exists(predict_output_dir):
    os.makedirs(predict_output_dir)

for mapname, point_symbol in model_dict.items():

    selected_model_weights = point_symbol
    map_input_dir_root = os.path.join(input_dir_root, mapname) + "/"
    print("=== Running a model prediction module ===")
    predict_img_patches(map_input_dir_root, model_weights_dir, selected_model_weights, predict_output_dir)

    stitch_output_dir = os.path.join(output_dir_root, 'stitch')
    if not os.path.isdir(stitch_output_dir):
        os.mkdir(stitch_output_dir)

    print("=== Running a stitching module ===")
    print("map_input_dir_root" + str(map_input_dir_root))
    stitch_to_single_result(map_input_dir_root, predict_output_dir, stitch_output_dir, crop_shift_size=1000)

# Set the directory containing your GeoJSON files
remove_pnts_from_legend(metadata_path,stitch_output_dir,visual_flag=False)

raster_output_path = os.path.join(output_dir_root, 'raster_layer')
if not os.path.exists(raster_output_path):
    os.makedirs(raster_output_path)

# geojson_to_raster(metadata_path, stitch_output_dir, raster_output_path)

end_time = time.time()
total_time = end_time - start_time
print(f"Total execution time: {total_time} seconds")
