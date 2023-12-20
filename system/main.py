import yaml
import json
import subprocess
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--config',
                    default='./config.yaml',
                    help='config file (.yml) containing the parameters to run the system.')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def run_command(command):
    # Run the command and capture the output
    process = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Get the standard output and error
    output = process.stdout
    error = process.stderr
    # Print the results
    print(output)
    if error:
        print(error)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    
    # ===============================================================
    # map layout analysis
    
    map_segment_command = f"python ./segmentation/map_area_segmentation.py --input_path {config.MAP_SEGMENT.INPUT_PATH} --binary_output_path {config.MAP_SEGMENT.BINARY_OUTPUT_PATH} --json_output_path {config.MAP_SEGMENT.JSON_OUTPUT_PATH} --intermediate_dir {config.MAP_SEGMENT.INTERMEDIATE_DIR}"
    
    run_command(map_segment_command)
    
    legend_segment_command = f"python3 -m pipelines.segmentation.run_pipeline --input {config.MAP_LEGEND_SEGMENT.INPUT_DIR} --output {config.MAP_LEGEND_SEGMENT.OUTPUT_DIR} --workdir {config.MAP_LEGEND_SEGMENT.INTERMEDIATE_DIR} --model {config.MAP_LEGEND_SEGMENT.MODEL_PATH}"
    
    run_command(legend_segment_command)
    
    legend_item_segment_command = f"python ./legend_item_segmentation/map_legend_segmentation.py --input_image {config.LEGEND_ITEM_SEGMENT.INPUT_PATH} --output_dir {config.LEGEND_ITEM_SEGMENT.OUTPUT_DIR} --preprocessing_for_cropping {config.LEGEND_ITEM_SEGMENT.PREPROCESSING_FOR_CROPPING} --postprocessing_for_crs {config.LEGEND_ITEM_SEGMENT.POSTPROCESSING_FOR_CRS} --path_to_mapkurator_output {config.LEGEND_ITEM_SEGMENT.MAPKURATOR_PATH} --path_to_intermediate {config.LEGEND_ITEM_SEGMENT.INTERMEDIATE_DIR}"
    
    run_command(legend_item_segment_command)
    
    legend_item_description_extract_command = f"python layout_segment_gpt4/gpt4_main.py --map_dir {config.LEGEND_ITEM_DESCRIPTION_EXTRACT.MAP_DIR} --legend_json_path {config.MAP_LEGEND_SEGMENT.OUTPUT_DIR} --symbol_json_dir {config.LEGEND_ITEM_SEGMENT.OUTPUT_DIR} --map_name {config.MAP_NAME} --gpt4_input_dir {config.LEGEND_ITEM_DESCRIPTION_EXTRACT.GPT_INPUT_DIR} --gpt4_output_dir {config.LEGEND_ITEM_DESCRIPTION_EXTRACT.GPT_OUTPUT_DIR} --gpt4_intermediate_dir {config.LEGEND_ITEM_DESCRIPTION_EXTRACT.INTERMEDIATE_DIR}"
    
    run_command(legend_item_description_extract_command)
    
    # ===============================================================
    # point/line/polygon extraction
    
    map_crop_command = f"python image_crop/map2patch.py --input_dir {config.CROP_IMAGE_GENERATION.MAP_DIR} --map_name {config.MAP_NAME} --patch_sizes {config.CROP_IMAGE_GENERATION.PATCH_SIZES} --strides {config.CROP_IMAGE_GENERATION.STRIDES} --output_dir {config.CROP_IMAGE_GENERATION.OUTPUT_DIR}"
    
    run_command(map_crop_command)
    
    line_extract_command = f"python -W ignore ./line/test_maps_from_folder.py --config {config.LINE_EXTRACTION.CONFIG} --checkpoint {config.LINE_EXTRACTION.CHECKPOINT} --map_name {config.MAP_NAME} --predict_raster {config.LINE_EXTRACTION.PREDICT_RASTER} --predict_vector {config.LINE_EXTRACTION.PREDICT_VECTOR} --line_feature_name {config.LINE_EXTRACTION.LINE_FEATURE_NAME} --test_tif_map_dir {config.LINE_EXTRACTION.TEST_TIF_MAP_DIR} --test_png_map_dir {config.LINE_EXTRACTION.TEST_PNG_MAP_DIR} --cropped_image_dir {config.CROP_IMAGE_GENERATION.OUTPUT_DIR} --map_bound_dir {config.LINE_EXTRACTION.MAP_BOUND_DIR} --prediction_dir {config.LINE_EXTRACTION.PREDICTION_DIR}"
    
    run_command(line_extract_command)
    
    
    
