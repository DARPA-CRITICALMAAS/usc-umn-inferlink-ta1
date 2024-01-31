
import os

import metadata_preprocessing
import metadata_postprocessing
import loam_inference
import polygon_output_handler

import sys

cwd_flag = False

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(args.log, "a")

        def write(self, message):
            self.terminal.write(message)
            try:
                self.log.write(message)
            except:
                self.log.write('\n  Unable to write to log file due to encoding issues...')
                print('\n  Unable to write to log file due to encoding issues...')

        def flush(self):
            pass

    sys.stdout = Logger()


    input_tif = args.path_to_tif
    input_json = args.path_to_json
    input_bound = args.path_to_bound
    dir_to_intermediate = args.dir_to_intermediate
    if '\\' not in dir_to_intermediate[-1:] and '/' not in dir_to_intermediate[-1:]:
        dir_to_intermediate = dir_to_intermediate+'/'
    #dir_to_solution = args.dir_to_solution
    dir_to_groundtruth = args.dir_to_groundtruth
    set_json = str_to_bool(args.set_json)
    map_preprocessing = str_to_bool(args.map_area_segmentation)
    performance_evaluation = str_to_bool(args.performance_evaluation)

    path_to_legend_solution = args.path_to_legend_solution
    dir_to_integrated_output = args.dir_to_integrated_output

    os.makedirs(os.path.dirname(dir_to_integrated_output), exist_ok=True)
    os.makedirs(os.path.dirname(dir_to_intermediate), exist_ok=True)

    if map_preprocessing==False:
        if os.path.isfile(input_bound) == False:
            print('The geojson file for area segmentation is not found, will proceed with area segmentation...')
            map_preprocessing = True

    #if set_json == False or os.path.isfile(input_json) == False:
        #input_json = path_to_legend_solution
    if set_json == False:
        input_json = path_to_legend_solution
    if os.path.isfile(input_json) == False:
        print('Please provide the json file from legend-item segmentation...')
        exit(1)
    if os.path.isfile(path_to_legend_solution) == False:
        print('Please provide the geojson file from legend-item segmentation...')
        exit(1)
        
    
    
    metadata_preprocessing.metadata_preprocessing(
        input_path_to_tif = input_tif,
        input_path_to_json = input_json,
        input_path_to_bound = input_bound,
        input_dir_to_intermediate = dir_to_intermediate,
        input_map_preprocessing = map_preprocessing,
    )


    metadata_postprocessing.metadata_postprocessing(
        input_path_to_tif = input_tif,
        input_path_to_json = input_json,
        input_dir_to_intermediate = dir_to_intermediate,
        input_dir_to_groundtruth = dir_to_groundtruth,
        input_performance_evaluation = performance_evaluation,
        crop_size=256
    )
    
    
    loam_inference.loam_inference(
        input_filtering_new_dataset = True,
        input_filtering_threshold = 0.33,
        input_k_fold_testing = 1,
        input_crop_size = 256,
        input_separate_validating_set = False,
        input_reading_predefined_testing = True,
        input_training_needed = False,
        input_targeted_map_file = 'targeted_map.csv',
        input_path_to_tif = input_tif,
        input_groundtruth_dir = dir_to_groundtruth,
        input_performance_evaluation = performance_evaluation
    )


    if os.path.isfile(path_to_legend_solution) == True:
        polygon_output_handler.output_handler(
            input_path_to_tif = input_tif,
            input_path_to_legend_solution = path_to_legend_solution,
            input_path_to_groundtruth_legend = input_json,
            input_dir_to_raster_polygon = 'LOAM_Intermediate/predict/cma/',
            input_dir_to_integrated_output = dir_to_integrated_output
        )
    
    exit(0)
    




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_tif', type=str, default='Input_Data/RI_Uxbridge.tif')
    parser.add_argument('--path_to_json', type=str, default='')
    #parser.add_argument('--path_to_json', type=str, default='Input_Data/RI_Uxbridge.json')
    parser.add_argument('--path_to_bound', type=str, default='')
    #parser.add_argument('--path_to_bound', type=str, default='Input_Data/RI_Uxbridge_expected_crop_region.geojson')

    parser.add_argument('--path_to_legend_solution', type=str, default='')
    parser.add_argument('--dir_to_integrated_output', type=str, default='Vectorization_Output')

    parser.add_argument('--dir_to_intermediate', type=str, default='LOAM_Intermediate/Metadata_Preprocessing/')
    parser.add_argument('--dir_to_groundtruth', type=str, default='Data/validation_groundtruth')

    parser.add_argument('--set_json', type=str, default='True')
    parser.add_argument('--map_area_segmentation', type=str, default='False')
    parser.add_argument('--performance_evaluation', type=str, default='False')

    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--log', type=str, default='log_file.txt')


    # python loam_handler.py --path_to_tif Input_Data/RI_Uxbridge.tif --path_to_json Input_Data/RI_Uxbridge.json --map_area_segmentation True --performance_evaluation False
    # python loam_handler.py --path_to_tif Input_Data/RI_Uxbridge.tif --path_to_legend_solution Input_Data/RI_Uxbridge_PolygonType.geojson
    

    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()
