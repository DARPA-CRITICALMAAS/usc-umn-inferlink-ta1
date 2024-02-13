
import os

import metadata_preprocessing
import metadata_postprocessing
import loam_inference
import polygon_output_handler

import sys
import json

from datetime import datetime

cwd_flag = False

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



def batch_main():
    this_map_count = 0
    total_map_count = len(os.listdir('Data/Nickel_MN/'))
    for target_map_cand in os.listdir('Data/Nickel_MN/'):
        if '.tif' in target_map_cand:
            #print(str(this_map_count) + '/' + str(total_map_count))
            this_map_count += 1

            runningtime_start = datetime.now()
            target_map = target_map_cand.split('.tif')[0]

            path_to_log = 'Nickel_MN_Output/'+str(target_map)+'.log'
            os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
            class Logger(object):
                def __init__(self):
                    self.terminal = sys.stdout
                    self.log = open(path_to_log, "a")

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

            input_tif = 'Data/Nickel_MN/'+target_map_cand
            input_json = 'Legend_Item_Segmentation/Nickel_MN_Output/Vectorization_Output/'+str(target_map)+'_PolygonType_internal.json'
            input_bound = 'x'
            dir_to_intermediate = 'Nickel_MN_Output'
            if '\\' not in dir_to_intermediate[-1:] and '/' not in dir_to_intermediate[-1:]:
                dir_to_intermediate = dir_to_intermediate+'/'
            dir_to_intermediate_preprocessing = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/Metadata_Preprocessing/')
            #dir_to_solution = args.dir_to_solution
            dir_to_groundtruth = 'x'
            set_json = True
            map_preprocessing = True
            performance_evaluation = False

            path_to_legend_solution = 'Legend_Item_Segmentation/Nickel_MN_Output/Vectorization_Output/'+str(target_map)+'_PolygonType.geojson'
            path_to_legend_description = 'Legend_Description_Extraction/nickel_mn_site/'+str(target_map)+'_polygon.json'
            dir_to_integrated_output = 'Nickel_MN_Output/Vectorization_Output'
            input_path_to_model = 'checkpoints/checkpoint_epoch14.pth'
            
            try:
                input_threads = 8
            except:
                print('Please input a valid number for number of threads in multi-processing...')
                exit(1)
            

            os.makedirs(os.path.dirname(dir_to_integrated_output), exist_ok=True)
            os.makedirs(os.path.dirname(dir_to_intermediate), exist_ok=True)
            os.makedirs(os.path.dirname(dir_to_intermediate_preprocessing), exist_ok=True)
            
            path_to_checkpoints = os.path.join(dir_to_intermediate, 'checkpoints/')
            os.makedirs(path_to_checkpoints, exist_ok=True)

            with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'w') as file:
                file.write('14')
                file.close()
            with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'r') as file:
                read_targeted_epoch = int(file.read().replace('\n', ''))
                file.close()


            if map_preprocessing==False:
                if os.path.isfile(input_bound) == False:
                    #print('The geojson file for area segmentation is not found, will proceed with area segmentation...')
                    map_preprocessing = True

            if set_json == False:
                input_json = path_to_legend_solution
            if os.path.isfile(input_json) == False or '.json' not in input_json:
                print('Please provide the json file from legend-item segmentation for bounding box...')
                with open(os.path.join('error_info.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue
            if os.path.isfile(path_to_legend_solution) == False or '.geojson' not in path_to_legend_solution:
                print('Please provide the geojson file from legend-item segmentation for schema format...')
                with open(os.path.join('error_info.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue
            #if os.path.isfile(path_to_legend_description) == False:
                #print('Please provide the json file from legend-item description for schema content...')
                #exit(1)
                
            
            legend_counter = poly_size_check(input_json)
            #print(legend_counter)
            if legend_counter == 0:
                print('There is zero poly legend items in this map...')

                with open(os.path.join('zero_poly.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue

            metadata_preprocessing.metadata_preprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_path_to_bound = input_bound,
                input_dir_to_intermediate = dir_to_intermediate_preprocessing,
                input_map_preprocessing = map_preprocessing,
                input_thread = input_threads,
                input_fast_processing = False
            )


            metadata_postprocessing.metadata_postprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_dir_to_intermediate = dir_to_intermediate,
                input_dir_to_groundtruth = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                crop_size=1024,
                input_thread = input_threads
            )
            
            
            loam_inference.loam_inference(
                input_filtering_new_dataset = True,
                input_filtering_threshold = 0.33,
                input_k_fold_testing = 1,
                input_crop_size = 1024,
                input_separate_validating_set = False,
                input_reading_predefined_testing = True,
                input_training_needed = False,
                input_dir_to_intermediate = dir_to_intermediate,
                input_targeted_map_file = 'targeted_map.csv',
                input_path_to_tif = input_tif,
                input_groundtruth_dir = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                input_thread = input_threads
            )


            if os.path.isfile(path_to_legend_solution) == True:
                polygon_output_handler.output_handler(
                    input_path_to_tif = input_tif,
                    input_path_to_legend_solution = path_to_legend_solution,
                    input_path_to_legend_description = path_to_legend_description,
                    input_path_to_json = input_json,
                    input_dir_to_raster_polygon = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/predict/cma/'),
                    input_dir_to_integrated_output = dir_to_integrated_output
                )
            

            with open(os.path.join('running_time_record_v0.csv'), 'a') as filev:
                filev.write(str(input_tif)+','+str(datetime.now()-runningtime_start)+'\n')
                filev.close()
            





def batch_main_v1():
    this_map_count = 0
    total_map_count = len(os.listdir('Data/New_Magmatic_Nickel/'))
    for target_map_cand in os.listdir('Data/New_Magmatic_Nickel/'):
        if '.tif' in target_map_cand:
            #print(str(this_map_count) + '/' + str(total_map_count))
            this_map_count += 1

            runningtime_start = datetime.now()
            target_map = target_map_cand.split('.tif')[0]

            path_to_log = 'Nickel_New_Output/'+str(target_map)+'.log'
            os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
            class Logger(object):
                def __init__(self):
                    self.terminal = sys.stdout
                    self.log = open(path_to_log, "a")

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

            input_tif = 'Data/New_Magmatic_Nickel/'+target_map_cand
            input_json = 'Legend_Item_Segmentation/Nickel_New_Output/Vectorization_Output/'+str(target_map)+'_PolygonType_internal.json'
            input_bound = 'x'
            dir_to_intermediate = 'Nickel_New_Output'
            if '\\' not in dir_to_intermediate[-1:] and '/' not in dir_to_intermediate[-1:]:
                dir_to_intermediate = dir_to_intermediate+'/'
            dir_to_intermediate_preprocessing = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/Metadata_Preprocessing/')
            #dir_to_solution = args.dir_to_solution
            dir_to_groundtruth = 'x'
            set_json = True
            map_preprocessing = True
            performance_evaluation = False

            path_to_legend_solution = 'Legend_Item_Segmentation/Nickel_New_Output/Vectorization_Output/'+str(target_map)+'_PolygonType.geojson'
            path_to_legend_description = 'Legend_Description_Extraction/new_nickel_site/'+str(target_map)+'_polygon.json'
            dir_to_integrated_output = 'Nickel_New_Output/Vectorization_Output'
            input_path_to_model = 'checkpoints/checkpoint_epoch14.pth'
            
            try:
                input_threads = 8
            except:
                print('Please input a valid number for number of threads in multi-processing...')
                exit(1)
            

            os.makedirs(os.path.dirname(dir_to_integrated_output), exist_ok=True)
            os.makedirs(os.path.dirname(dir_to_intermediate), exist_ok=True)
            os.makedirs(os.path.dirname(dir_to_intermediate_preprocessing), exist_ok=True)
            
            path_to_checkpoints = os.path.join(dir_to_intermediate, 'checkpoints/')
            os.makedirs(path_to_checkpoints, exist_ok=True)

            with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'w') as file:
                file.write('14')
                file.close()
            with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'r') as file:
                read_targeted_epoch = int(file.read().replace('\n', ''))
                file.close()
            

            if map_preprocessing==False:
                if os.path.isfile(input_bound) == False:
                    #print('The geojson file for area segmentation is not found, will proceed with area segmentation...')
                    map_preprocessing = True

            if set_json == False:
                input_json = path_to_legend_solution
            if os.path.isfile(input_json) == False or '.json' not in input_json:
                print('Please provide the json file from legend-item segmentation for bounding box...')
                with open(os.path.join('error_info.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue
            if os.path.isfile(path_to_legend_solution) == False or '.geojson' not in path_to_legend_solution:
                print('Please provide the geojson file from legend-item segmentation for schema format...')
                with open(os.path.join('error_info.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue
            #if os.path.isfile(path_to_legend_description) == False:
                #print('Please provide the json file from legend-item description for schema content...')
                #exit(1)
                
            
            legend_counter = poly_size_check(input_json)
            #print(legend_counter)
            if legend_counter == 0:
                print('There is zero poly legend items in this map...')

                with open(os.path.join('zero_poly.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue


            metadata_preprocessing.metadata_preprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_path_to_bound = input_bound,
                input_dir_to_intermediate = dir_to_intermediate_preprocessing,
                input_map_preprocessing = map_preprocessing,
                input_thread = input_threads,
                input_fast_processing = True
            )


            metadata_postprocessing.metadata_postprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_dir_to_intermediate = dir_to_intermediate,
                input_dir_to_groundtruth = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                crop_size=1024,
                input_thread = input_threads
            )
            
            
            loam_inference.loam_inference(
                input_filtering_new_dataset = True,
                input_filtering_threshold = 0.33,
                input_k_fold_testing = 1,
                input_crop_size = 1024,
                input_separate_validating_set = False,
                input_reading_predefined_testing = True,
                input_training_needed = False,
                input_dir_to_intermediate = dir_to_intermediate,
                input_targeted_map_file = 'targeted_map.csv',
                input_path_to_tif = input_tif,
                input_groundtruth_dir = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                input_thread = input_threads
            )


            if os.path.isfile(path_to_legend_solution) == True:
                polygon_output_handler.output_handler(
                    input_path_to_tif = input_tif,
                    input_path_to_legend_solution = path_to_legend_solution,
                    input_path_to_legend_description = path_to_legend_description,
                    input_path_to_json = input_json,
                    input_dir_to_raster_polygon = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/predict/cma/'),
                    input_dir_to_integrated_output = dir_to_integrated_output
                )
            

            with open(os.path.join('running_time_record_v0.csv'), 'a') as filev:
                filev.write(str(input_tif)+','+str(datetime.now()-runningtime_start)+'\n')
                filev.close()
            




def batch_main_v0():
    this_map_count = 0
    total_map_count = len(os.listdir('Data/Magmatic_Nickel/'))
    for target_map_cand in os.listdir('Data/Magmatic_Nickel/'):
        if '.tif' in target_map_cand:
            runningtime_start = datetime.now()
            target_map = target_map_cand.split('.tif')[0]

            path_to_log = 'Nickel_Output/'+str(target_map)+'.log'
            os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
            class Logger(object):
                def __init__(self):
                    self.terminal = sys.stdout
                    self.log = open(path_to_log, "a")

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

            input_tif = 'Data/Magmatic_Nickel/'+target_map_cand
            input_json = 'Legend_Item_Segmentation/Nickel_Output/Vectorization_Output/'+str(target_map)+'_PolygonType_internal.json'
            input_bound = 'x'
            dir_to_intermediate = 'Nickel_Output'
            if '\\' not in dir_to_intermediate[-1:] and '/' not in dir_to_intermediate[-1:]:
                dir_to_intermediate = dir_to_intermediate+'/'
            dir_to_intermediate_preprocessing = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/Metadata_Preprocessing/')
            #dir_to_solution = args.dir_to_solution
            dir_to_groundtruth = 'x'
            set_json = True
            map_preprocessing = True
            performance_evaluation = False

            path_to_legend_solution = 'Legend_Item_Segmentation/Nickel_Output/Vectorization_Output/'+str(target_map)+'_PolygonType.geojson'
            path_to_legend_description = 'Legend_Description_Extraction/nickel_site/'+str(target_map)+'_polygon.json'
            dir_to_integrated_output = 'Nickel_Output/Vectorization_Output'
            input_path_to_model = 'checkpoints/checkpoint_epoch14.pth'
            
            try:
                input_threads = 8
            except:
                print('Please input a valid number for number of threads in multi-processing...')
                exit(1)
            

            os.makedirs(os.path.dirname(dir_to_integrated_output), exist_ok=True)
            os.makedirs(os.path.dirname(dir_to_intermediate), exist_ok=True)
            os.makedirs(os.path.dirname(dir_to_intermediate_preprocessing), exist_ok=True)
            
            path_to_checkpoints = os.path.join(dir_to_intermediate, 'checkpoints/')
            os.makedirs(path_to_checkpoints, exist_ok=True)

            with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'w') as file:
                file.write('14')
                file.close()
            with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'r') as file:
                read_targeted_epoch = int(file.read().replace('\n', ''))
                file.close()
            

            if map_preprocessing==False:
                if os.path.isfile(input_bound) == False:
                    #print('The geojson file for area segmentation is not found, will proceed with area segmentation...')
                    map_preprocessing = True

            if set_json == False:
                input_json = path_to_legend_solution
            if os.path.isfile(input_json) == False or '.json' not in input_json:
                print('Please provide the json file from legend-item segmentation for bounding box...')
                with open(os.path.join('error_info.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue
            if os.path.isfile(path_to_legend_solution) == False or '.geojson' not in path_to_legend_solution:
                print('Please provide the geojson file from legend-item segmentation for schema format...')
                with open(os.path.join('error_info.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue
            #if os.path.isfile(path_to_legend_description) == False:
                #print('Please provide the json file from legend-item description for schema content...')
                #exit(1)
                
            
            legend_counter = poly_size_check(input_json)
            #print(legend_counter)
            if legend_counter == 0:
                print('There is zero poly legend items in this map...')

                with open(os.path.join('zero_poly.txt'), 'a') as filef:
                    filef.write(str(input_tif)+'\n')
                    filef.close()
                continue


            metadata_preprocessing.metadata_preprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_path_to_bound = input_bound,
                input_dir_to_intermediate = dir_to_intermediate_preprocessing,
                input_map_preprocessing = map_preprocessing,
                input_thread = input_threads,
                input_fast_processing = True
            )


            metadata_postprocessing.metadata_postprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_dir_to_intermediate = dir_to_intermediate,
                input_dir_to_groundtruth = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                crop_size=1024,
                input_thread = input_threads
            )
            
            
            loam_inference.loam_inference(
                input_filtering_new_dataset = True,
                input_filtering_threshold = 0.33,
                input_k_fold_testing = 1,
                input_crop_size = 1024,
                input_separate_validating_set = False,
                input_reading_predefined_testing = True,
                input_training_needed = False,
                input_dir_to_intermediate = dir_to_intermediate,
                input_targeted_map_file = 'targeted_map.csv',
                input_path_to_tif = input_tif,
                input_groundtruth_dir = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                input_thread = input_threads
            )


            if os.path.isfile(path_to_legend_solution) == True:
                polygon_output_handler.output_handler(
                    input_path_to_tif = input_tif,
                    input_path_to_legend_solution = path_to_legend_solution,
                    input_path_to_legend_description = path_to_legend_description,
                    input_path_to_json = input_json,
                    input_dir_to_raster_polygon = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/predict/cma/'),
                    input_dir_to_integrated_output = dir_to_integrated_output
                )
            

            with open(os.path.join('running_time_record_v0.csv'), 'a') as filev:
                filev.write(str(input_tif)+','+str(datetime.now()-runningtime_start)+'\n')
                filev.close()
            



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



def file_checker(input_tif, input_json, path_to_legend_solution, path_to_legend_description, input_path_to_model):
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
    if os.path.isfile(path_to_legend_solution) == False or '.geojson' not in path_to_legend_solution:
        print('Please provide the correct path to the geojson file from legend-item segmentation for schema format...')
        print('Current path:' + path_to_legend_solution)
        file_integrity = False
    if os.path.isfile(path_to_legend_description) == False:
        print('Please provide the correct path to the json file from legend-description extraction for schema content...')
        print('Current path:' + path_to_legend_description)
        file_integrity = False
    if os.path.isfile(input_path_to_model) == False or '.pth' not in input_path_to_model:
        print('Please provide the correct path to the pre-trained model...')
        print('Current path:' + input_path_to_model)
        file_integrity = False
    
    if file_integrity == False:
        print('Abort due to file integrity...')
        sys.exit(1)
    return


def sub_file_checker(this_section, dir_to_intermediate_preprocessing, dir_to_intermediate):
    file_integrity = True
    if this_section == 0:
        if os.path.isfile(os.path.join(dir_to_intermediate_preprocessing, 'LOAM_Intermediate/Metadata_Preprocessing', 'intermediate9/auxiliary_info.csv')) == False:
            print('Missing file... ' + str(os.path.join(dir_to_intermediate_preprocessing, 'LOAM_Intermediate/Metadata_Preprocessing', 'intermediate9/auxiliary_info.csv')))
            file_integrity = False
        if os.path.isfile(os.path.join(dir_to_intermediate_preprocessing, 'intermediate7_2', 'running_time_record_v3.csv')) == False:
            print('Missing file... ' + str(os.path.join(dir_to_intermediate_preprocessing, 'intermediate7_2', 'running_time_record_v3.csv')))
            file_integrity = False
    if this_section == 1:
        if os.path.isfile(os.path.join(dir_to_intermediate, 'LOAM_Intermediate', 'data', 'running_time_record_v1.csv')) == False:
            print('Missing file... ' + str(os.path.join(dir_to_intermediate, 'LOAM_Intermediate', 'data', 'running_time_record_v1.csv')))
            file_integrity = False
    if this_section == 2:
        if os.path.isdir(os.path.join(dir_to_intermediate, 'LOAM_Intermediate/predict/cma/')) == False:
            print('Missing dir... ' + str(os.path.join(dir_to_intermediate, 'LOAM_Intermediate/predict/cma/')))
            file_integrity = False

    if file_integrity == False:
        print('Abort due to file integrity...')
        sys.exit(1)
    print('Should be good to go...')
    return


def main():
    global_runningtime_start = datetime.now()
    hackathon_batch = str_to_bool(args.hackathon)
    if hackathon_batch == True:
        batch_main()
        return

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

    argument_checker()


    input_tif = args.path_to_tif
    input_json = args.path_to_json
    input_bound = args.path_to_bound
    dir_to_intermediate = args.dir_to_intermediate
    if '\\' not in dir_to_intermediate[-1:] and '/' not in dir_to_intermediate[-1:]:
        dir_to_intermediate = dir_to_intermediate+'/'
    dir_to_intermediate_preprocessing = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/Metadata_Preprocessing/')
    #dir_to_solution = args.dir_to_solution
    dir_to_groundtruth = args.dir_to_groundtruth
    set_json = str_to_bool(args.set_json)
    map_preprocessing = str_to_bool(args.map_area_segmentation)
    performance_evaluation = str_to_bool(args.performance_evaluation)

    path_to_legend_solution = args.path_to_legend_solution
    path_to_legend_description = args.path_to_legend_description
    dir_to_integrated_output = args.dir_to_integrated_output
    input_path_to_model = args.path_to_model
    
    try:
        input_threads = int(args.threads)
    except:
        print('Please input a valid number for number of threads in multi-processing...')
        sys.exit(1)
    

    os.makedirs(os.path.dirname(dir_to_integrated_output), exist_ok=True)
    os.makedirs(os.path.dirname(dir_to_intermediate), exist_ok=True)
    os.makedirs(os.path.dirname(dir_to_intermediate_preprocessing), exist_ok=True)
    
    path_to_checkpoints = os.path.join(dir_to_intermediate, 'checkpoints')
    os.makedirs(path_to_checkpoints, exist_ok=True)

    with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'w') as file:
        file.write('14')
        file.close()
    #with open(os.path.join(dir_to_intermediate, 'checkpoints/epoch.txt'), 'r') as file:
        #read_targeted_epoch = int(file.read().replace('\n', ''))
        #file.close()
    

    if map_preprocessing==False:
        if os.path.isfile(input_bound) == False:
            #print('The geojson file for area segmentation is not found, will proceed with area segmentation...')
            map_preprocessing = True

    if set_json == False:
        input_json = path_to_legend_solution
    
    file_checker(input_tif, input_json, path_to_legend_solution, path_to_legend_description, input_path_to_model)
    
    legend_counter = poly_size_check(input_json)
    if legend_counter == 0:
        print('There is zero poly legend items in this map...')
        sys.exit(0)
    
    this_testing = str_to_bool(args.testing)
    this_testing_section = args.testing_section
    if this_testing == False:
        metadata_preprocessing.metadata_preprocessing(
            input_path_to_tif = input_tif,
            input_path_to_json = input_json,
            input_path_to_bound = input_bound,
            input_dir_to_intermediate = dir_to_intermediate_preprocessing,
            input_map_preprocessing = map_preprocessing,
            input_thread = input_threads
        )

        metadata_postprocessing.metadata_postprocessing(
            input_path_to_tif = input_tif,
            input_path_to_json = input_json,
            input_dir_to_intermediate = dir_to_intermediate,
            input_dir_to_groundtruth = dir_to_groundtruth,
            input_performance_evaluation = performance_evaluation,
            crop_size=1024,
            input_thread = input_threads
        )
        
        loam_inference.loam_inference(
            input_filtering_new_dataset = True,
            input_filtering_threshold = 0.33,
            input_k_fold_testing = 1,
            input_crop_size = 1024,
            input_separate_validating_set = False,
            input_reading_predefined_testing = True,
            input_training_needed = False,
            input_dir_to_intermediate = dir_to_intermediate,
            input_targeted_map_file = 'targeted_map.csv',
            input_path_to_tif = input_tif,
            input_groundtruth_dir = dir_to_groundtruth,
            input_performance_evaluation = performance_evaluation,
            input_thread = input_threads,
            input_path_to_model = input_path_to_model
        )

        polygon_output_handler.output_handler(
            input_path_to_tif = input_tif,
            input_path_to_legend_solution = path_to_legend_solution,
            input_path_to_legend_description = path_to_legend_description,
            input_path_to_json = input_json,
            input_dir_to_raster_polygon = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/predict/cma/'),
            input_dir_to_integrated_output = dir_to_integrated_output
        )
    else:
        print('You are going to run only some components of the module with section(s): ' + str(this_testing_section))
        print('In the process of making sure you have files for latter parts ready...')

        if '0' not in this_testing_section and '1' in this_testing_section:
            sub_file_checker(0, dir_to_intermediate_preprocessing, None)
        if '1' not in this_testing_section and '2' in this_testing_section:
            sub_file_checker(1, None, dir_to_intermediate)
        if '2' not in this_testing_section and '3' in this_testing_section:
            sub_file_checker(2, None, dir_to_intermediate)


        if '0' in this_testing_section:
            metadata_preprocessing.metadata_preprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_path_to_bound = input_bound,
                input_dir_to_intermediate = dir_to_intermediate_preprocessing,
                input_map_preprocessing = map_preprocessing,
                input_thread = input_threads
            )

        if '1' in this_testing_section:
            metadata_postprocessing.metadata_postprocessing(
                input_path_to_tif = input_tif,
                input_path_to_json = input_json,
                input_dir_to_intermediate = dir_to_intermediate,
                input_dir_to_groundtruth = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                crop_size=1024,
                input_thread = input_threads
            )
            
        if '2' in this_testing_section:
            loam_inference.loam_inference(
                input_filtering_new_dataset = True,
                input_filtering_threshold = 0.33,
                input_k_fold_testing = 1,
                input_crop_size = 1024,
                input_separate_validating_set = False,
                input_reading_predefined_testing = True,
                input_training_needed = False,
                input_dir_to_intermediate = dir_to_intermediate,
                input_targeted_map_file = 'targeted_map.csv',
                input_path_to_tif = input_tif,
                input_groundtruth_dir = dir_to_groundtruth,
                input_performance_evaluation = performance_evaluation,
                input_thread = input_threads,
                input_path_to_model = input_path_to_model
            )

        if '3' in this_testing_section:
            polygon_output_handler.output_handler(
                input_path_to_tif = input_tif,
                input_path_to_legend_solution = path_to_legend_solution,
                input_path_to_legend_description = path_to_legend_description,
                input_path_to_json = input_json,
                input_dir_to_raster_polygon = os.path.join(dir_to_intermediate, 'LOAM_Intermediate/predict/cma/'),
                input_dir_to_integrated_output = dir_to_integrated_output
            )


    
    print('Overall processing time: '+str(datetime.now()-global_runningtime_start))
    sys.exit(0)
    




import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_tif', type=str, default='')
    #parser.add_argument('--path_to_tif', type=str, default='Input_Data/RI_Uxbridge.tif')
    parser.add_argument('--path_to_json', type=str, default='')
    #parser.add_argument('--path_to_json', type=str, default='Input_Data/RI_Uxbridge.json')
    parser.add_argument('--path_to_bound', type=str, default='')
    #parser.add_argument('--path_to_bound', type=str, default='Input_Data/RI_Uxbridge_expected_crop_region.geojson')

    parser.add_argument('--path_to_legend_solution', type=str, default='')
    parser.add_argument('--path_to_legend_description', type=str, default='')
    parser.add_argument('--dir_to_integrated_output', type=str, default='Vectorization_Output')

    parser.add_argument('--dir_to_intermediate', type=str, default='Example_Output') # default='LOAM_Intermediate/Metadata_Preprocessing/'
    parser.add_argument('--dir_to_groundtruth', type=str, default='Data/validation_groundtruth')

    parser.add_argument('--set_json', type=str, default='True')
    parser.add_argument('--map_area_segmentation', type=str, default='False')
    parser.add_argument('--performance_evaluation', type=str, default='False')

    parser.add_argument('--threads', type=str, default='10')

    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--log', type=str, default='log_file.txt')

    parser.add_argument('--path_to_model', type=str, default='checkpoints/checkpoint_epoch14.pth')

    parser.add_argument('--hackathon', type=str, default='False')

    parser.add_argument('--testing', type=str, default='False')
    parser.add_argument('--testing_section', type=str, default='0')


    # python loam_handler.py --path_to_tif Input_Data/RI_Uxbridge.tif --path_to_json Input_Data/RI_Uxbridge.json --map_area_segmentation True --performance_evaluation False
    # python loam_handler.py --path_to_tif Input_Data/RI_Uxbridge.tif --path_to_legend_solution Input_Data/RI_Uxbridge_PolygonType.geojson
    

    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()
