
import os

import permian_prototype
import polygon_output_handler

import sys
import json

from datetime import datetime
import threading
import time

import logging
import traceback


def impulse(global_runningtime_start):
    while True:
        print('......... I have been still running after you started the polygon-extraction module for ... '+str(datetime.now()-global_runningtime_start))
        time.sleep(30.0)



def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def poly_size_check(path_to_json):
    legend_counter = 0
    with open(path_to_json) as f:
        gj = json.load(f)
    for this_gj in gj['shapes']:
        #print(this_gj)
        names = this_gj['label']
        features = this_gj['points']

        if '_poly' in names:
            legend_counter += + 1
    
    return legend_counter





def argument_checker():
    # make sure that mandatory inputs are given...
    input_integrity = True
    if len(args.path_to_tif) == 0:
        print('"--path_to_tif" is a mandatory input (raster map)...')
        input_integrity = False
    if len(args.path_to_json) == 0:
        print('"--path_to_json" is a mandatory input (internal json output from legend-item segmentation)...')
        input_integrity = False
    if len(args.path_to_legend_solution) == 0:
        print('"--path_to_legend_solution" is a mandatory input (internal geojson output from legend-item segmentation)...')
        input_integrity = False
    if len(args.path_to_legend_description) == 0:
        print('"--path_to_legend_description" is a mandatory input (internal json output from legend-description extraction)...')
        input_integrity = False
    
    if input_integrity == False:
        print('Abort due to input integrity...')
        sys.exit(1)
    return



def file_checker(input_tif, input_json, path_to_legend_solution, path_to_legend_description, input_path_to_model, inpt_set_schema):
    # make sure that files needed are existing...
    file_integrity = True
    if os.path.isfile(input_tif) == False :
        print('Please provide the correct path to the tif file...')
        print('Current path:' + input_tif)
        file_integrity = False
    if os.path.isfile(input_json) == False or '.json' not in input_json:
        print('Please provide the correct path to the json file from legend-item segmentation for bounding box...')
        print('Current path:' + input_json)
        file_integrity = False
    if inpt_set_schema == True and (os.path.isfile(path_to_legend_solution) == False or '.json' not in input_json):
        print('Please provide the correct path to the geojson file from legend-item segmentation for schema format...')
        print('Current path:' + path_to_legend_solution)
        file_integrity = False
    if inpt_set_schema == True and (os.path.isfile(path_to_legend_description) == False):
        print('Please provide the correct path to the json file from legend-description extraction for schema content...')
        print('Current path:' + path_to_legend_description)
        file_integrity = False
    '''
    if os.path.isfile(input_path_to_model) == False or '.pth' not in input_path_to_model:
        print('Please provide the correct path to the pre-trained model...')
        print('Current path:' + input_path_to_model)
        file_integrity = False
    '''

    if file_integrity == False:
        print('Abort due to file integrity...')
        sys.exit(1)
    return


'''
import torch
import torch.nn as nn
from LOAM.loam_model import LOAM

def gpu_checker(input_allow_cpu):
    num_of_gpus = torch.cuda.device_count()
    
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print('Using device... ', device)
    
    netL = LOAM(n_channels=7, n_classes=2, bilinear=False).to(device)
    if device.type == 'cuda':
        if num_of_gpus > 1:
            netL = nn.DataParallel(netL, list(range(num_of_gpus)))
            print('Gain access to a total of '+str(num_of_gpus)+' GPUs...')
        else:
            print('Gain access to 1 GPU...')
        print('Please ensure that this module gains access to all available GPUs if you notice this line...')
        print('\n')
    if device.type == 'cpu':
        if input_allow_cpu == False:
            print('Abort due to gpu(s) accessibility...')
            print('Please set --allow_cpu True if you still want to proceed...')
            sys.exit(1)
        else:
            print('You are allowing the model to run only with CPU...')
            print('Please be aware of the possible long running time...')
            print('\n')
    return
'''



def main():
    global_runningtime_start = datetime.now()
    timer = threading.Timer(0.0, impulse, [global_runningtime_start])
    timer.daemon = True  # Set as a daemon so it will be killed once the main program exits
    timer.start()


    log_dir, log_name = os.path.split(args.log)
    os.makedirs(log_dir, exist_ok=True)

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

        
    logging.basicConfig(level=logging.ERROR, 
                        filename=args.log, 
                        filemode='a', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = handle_exception

    argument_checker()


    input_tif = args.path_to_tif
    input_json = args.path_to_json
    input_bound = args.path_to_bound
    dir_to_intermediate = args.dir_to_intermediate
    #dir_to_solution = args.dir_to_solution
    dir_to_groundtruth = args.dir_to_groundtruth
    set_json = str_to_bool(args.set_json)
    set_schema = str_to_bool(args.set_schema)
    map_preprocessing = str_to_bool(args.map_area_segmentation)
    performance_evaluation = str_to_bool(args.performance_evaluation)

    path_to_legend_solution = args.path_to_legend_solution
    path_to_legend_description = args.path_to_legend_description
    dir_to_vector_output = args.dir_to_vector_output
    input_path_to_model = args.path_to_model
    input_allow_cpu = str_to_bool(args.allow_cpu)

    efficiency_trade_off = 1
    if (args.trade_off).isdigit() == True:
        efficiency_trade_off = int(args.trade_off)
    if efficiency_trade_off > 1:
        efficiency_trade_off = 1

    dir_to_raster_output = args.dir_to_raster_output
    
    try:
        input_threads = int(args.threads)
    except:
        print('Please input a valid number for number of threads in multi-processing...')
        sys.exit(1)
    
    if set_json:
        path_to_legend_solution = 'placeholder.geojson'
        path_to_legend_description = 'placeholder.json'
    
    print('')
    print('')
    print('======================================================================================================')
    print('You are currently running polygon-extraction module with an efficiency trade-off of '+str(efficiency_trade_off)+' ...')
    print('...(0: higher accuracy with higher space complexity,  1: lower accuracy with lower space complexity)...')
    print('You are currently enabling multi-processing with a number of threads for '+str(input_threads)+' ...')
    print('======================================================================================================')
    print('')
    print('')
    

    os.makedirs(dir_to_vector_output, exist_ok=True)
    os.makedirs(dir_to_intermediate, exist_ok=True)
    os.makedirs(dir_to_raster_output, exist_ok=True)


    if map_preprocessing==False:
        if os.path.isfile(input_bound) == False:
            #print('The geojson file for area segmentation is not found, will proceed with area segmentation...')
            map_preprocessing = True

    
    file_checker(input_tif, input_json, path_to_legend_solution, path_to_legend_description, input_path_to_model, set_schema)
    print('File_checker passed...')
    #gpu_checker(input_allow_cpu)
    
    legend_counter = poly_size_check(input_json)
    if legend_counter == 0:
        print('There is zero poly legend items in this map...')
        path_list = input_tif.replace('\\','/').split('/')
        target_map_name = os.path.splitext(path_list[-1])[0]
        output_geo_path = os.path.join(dir_to_vector_output, target_map_name, target_map_name+'_empty.geojson')

        if not os.path.exists(dir_to_vector_output):
            os.makedirs(dir_to_vector_output)
        if not os.path.exists(os.path.join(dir_to_vector_output, target_map_name)):
            os.makedirs(os.path.join(dir_to_vector_output, target_map_name))
        if not os.path.exists(dir_to_raster_output):
            os.makedirs(dir_to_raster_output)
        if not os.path.exists(os.path.join(dir_to_raster_output, target_map_name)):
            os.makedirs(os.path.join(dir_to_raster_output, target_map_name))

        empty_geojson = {
            "type": "FeatureCollection",
            "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:EPSG::3857" } },
            "features": []
        }

        with open(output_geo_path, 'w') as f:
            json.dump(empty_geojson, f)
        sys.exit(0)
    else:
        path_list = input_tif.replace('\\','/').split('/')
        target_map_name = os.path.splitext(path_list[-1])[0]
        if not os.path.exists(dir_to_raster_output):
            os.makedirs(dir_to_raster_output)
        if not os.path.exists(os.path.join(dir_to_raster_output, target_map_name)):
            os.makedirs(os.path.join(dir_to_raster_output, target_map_name))

    this_testing = str_to_bool(args.testing)
    this_testing_section = args.testing_section

    if this_testing == False:
        permian_prototype.run_permian(
            dir_to_intermediate = dir_to_intermediate, 
            path_to_tif = input_tif, 
            path_to_json = input_json, 
            path_to_source_groundtruth = dir_to_groundtruth,
            input_dir_to_raster_output = os.path.join(dir_to_raster_output, target_map_name),
            input_thread = input_threads,
            input_efficiency_trade_off = efficiency_trade_off
        )

        polygon_output_handler.output_handler(
            input_path_to_tif = input_tif,
            input_path_to_legend_solution = path_to_legend_solution,
            input_path_to_legend_description = path_to_legend_description,
            input_path_to_json = input_json,
            input_dir_to_raster_polygon = os.path.join(dir_to_raster_output, target_map_name),
            input_dir_to_integrated_output = dir_to_vector_output,
            input_dir_to_raster_output = dir_to_raster_output,
            input_set_schema = set_schema,
            input_vectorization = True
        )
    else:
        print('Testing mode is not available in this version, as there are only 3 sections...')

    
    print('Overall processing time: '+str(datetime.now()-global_runningtime_start))
    #timer.cancel()
    sys.exit(0)
    




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_tif', type=str, default='')
    parser.add_argument('--path_to_json', type=str, default='')
    parser.add_argument('--path_to_bound', type=str, default='')

    parser.add_argument('--path_to_legend_solution', type=str, default='x')
    parser.add_argument('--path_to_legend_description', type=str, default='x')

    parser.add_argument('--dir_to_vector_output', type=str, default='Example_Output/PERMIAN_Vector_Output')
    parser.add_argument('--dir_to_raster_output', type=str, default='Example_Output/PERMIAN_Raster_Output')

    parser.add_argument('--dir_to_intermediate', type=str, default='Example_Output')
    parser.add_argument('--dir_to_groundtruth', type=str, default='x')

    parser.add_argument('--set_json', type=str, default='False')
    parser.add_argument('--set_schema', type=str, default='True')
    parser.add_argument('--map_area_segmentation', type=str, default='True')
    parser.add_argument('--performance_evaluation', type=str, default='False')

    parser.add_argument('--threads', type=str, default='10')
    parser.add_argument('--version', type=str, default='2')
    parser.add_argument('--log', type=str, default='Example_Output/log.log')

    parser.add_argument('--path_to_model', type=str, default='checkpoints/checkpoint_epoch14.pth')

    parser.add_argument('--testing', type=str, default='False')
    parser.add_argument('--testing_section', type=str, default='0')

    parser.add_argument('--allow_cpu', type=str, default='False')
    parser.add_argument('--trade_off', type=str, default='1')

    args = parser.parse_args()
    
    main()
