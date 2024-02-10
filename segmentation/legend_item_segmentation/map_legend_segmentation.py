
import os
import argparse
#import link_for_processing
import link_for_integrated_processing
import link_for_postprocessing

#import legend_item_vector_evaluation_v02

import csv
import sys

from datetime import datetime


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def batch_extract_main():
    runningtime_start = datetime.now()
    this_map_count = 0
    total_map_count = len(os.listdir('Data/nickel_new/'))
    for target_map_cand in os.listdir('Data/nickel_new/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/nickel_new/'+str(target_map)+'.tif'
            output_dir = 'Nickel_New_Output/Vectorization_Output/'
            path_to_intermediate = 'Nickel_New_Output/LINK_Intermediate/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Data/nickel_new_segmentation/'+str(target_map)+'_map_segmentation.json'
            #path_to_mapkurator_output = 'MapKurator/ta1-feature-evaluation/'+str(target_map)+'.geojson'
            path_to_mapkurator_output = 'None.geojson'

            if os.path.isfile(input_legend_segmentation) == False:
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                this_map_count += 1
                with open('missing.csv', 'a') as filev:
                    filev.write(str(target_map)+'\n')
                    filev.close()
                continue

            preprocessing_for_cropping = None
            postprocessing_for_crs = True
            competition_custom = False
            this_version = '1.2'

            path_to_log = 'Nickel_New_Output/'+str(target_map)+'.log'
            os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            class Logger(object):
                def __init__(self):
                    self.terminal = sys.stdout
                    self.log = open(path_to_log, 'a')
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
            
            
            flagging = link_for_integrated_processing.start_linking(target_map_name, input_image, None, path_to_intermediate, None, input_legend_segmentation, path_to_mapkurator_output, None, None, competition_custom, False, this_version)
            if flagging == True:
                link_for_postprocessing.start_linking_postprocessing(target_map, input_image, output_dir, path_to_intermediate, None, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom)
                print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count)+'   ...'+str(datetime.now()-runningtime_start))
            else:
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
            #start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True, '1.2')
            this_map_count += 1



def batch_main():
    missing_list = []
    with open('missing.csv', newline='') as fdd:
        reader = csv.reader(fdd)
        for row in reader:
            missing_list.append(row[0])
    print(missing_list)


    runningtime_start = datetime.now()
    this_map_count = 0
    total_map_count = int(len(os.listdir('Data/testing/'))/2)
    for target_map_cand in os.listdir('Data/testing/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]
            if target_map_cand in missing_list:
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                this_map_count += 1
                continue

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/testing/'+str(target_map)+'.tif'
            groundtruth_dir = 'Data/testing_groundtruth'
            output_dir = 'Example_Output/Vectorization_Output/'
            path_to_intermediate = 'Example_Output/LINK_Intermediate/testing/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-evaluation/'+str(target_map)+'.geojson'

            preprocessing_for_cropping = None
            postprocessing_for_crs = True
            competition_custom = True
            this_version = '1.2'

            path_to_log = 'Example_Output/'+str(target_map)+'.log'
            os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            class Logger(object):
                def __init__(self):
                    self.terminal = sys.stdout
                    self.log = open(path_to_log, 'a')
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
            
            
            flagging = link_for_integrated_processing.start_linking(target_map_name, input_image, None, path_to_intermediate, None, input_legend_segmentation, path_to_mapkurator_output, None, None, competition_custom, True, this_version)
            if flagging == True:
                link_for_postprocessing.start_linking_postprocessing(target_map, input_image, output_dir, path_to_intermediate, None, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom)
                print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count)+'   ...'+str(datetime.now()-runningtime_start))
                #legend_item_vector_evaluation_v02.single_evaluation(target_map_name, output_dir, groundtruth_dir, 'Example_Output', str(datetime.now()-runningtime_start))
            else:
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
            #start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True, '1.2')
            this_map_count += 1



    runningtime_start = datetime.now()
    this_map_count = 0
    total_map_count = int(len(os.listdir('Data/validation/'))/2)
    for target_map_cand in os.listdir('Data/validation/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]
            if target_map_cand in missing_list:
                this_map_count += 1
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                continue

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/validation/'+str(target_map)+'.tif'
            groundtruth_dir = 'Data/validation_groundtruth'
            path_to_intermediate = 'Example_Output/LINK_Intermediate/validation/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-validation/'+str(target_map)+'.geojson'

            path_to_log = 'Example_Output/'+str(target_map)+'.log'
            os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            class Logger(object):
                def __init__(self):
                    self.terminal = sys.stdout
                    self.log = open(path_to_log, 'a')
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

            
            flagging = link_for_integrated_processing.start_linking(target_map_name, input_image, None, path_to_intermediate, None, input_legend_segmentation, path_to_mapkurator_output, None, None, competition_custom, True, this_version)
            if flagging == True:
                link_for_postprocessing.start_linking_postprocessing(target_map, input_image, output_dir, path_to_intermediate, None, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom)
                print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count)+'   ...'+str(datetime.now()-runningtime_start))
                #legend_item_vector_evaluation_v02.single_evaluation(target_map_name, output_dir, groundtruth_dir, 'Example_Output', str(datetime.now()-runningtime_start))
            else:
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
            #start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True, '1.2')
            this_map_count += 1



    runningtime_start = datetime.now()
    this_map_count = 0
    total_map_count = int(len(os.listdir('Data/training/'))/2)
    for target_map_cand in os.listdir('Data/training/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]
            if target_map_cand in missing_list:
                this_map_count += 1
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                continue

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/training/'+str(target_map)+'.tif'
            groundtruth_dir = 'Data/training_groundtruth'
            path_to_intermediate = 'Example_Output/LINK_Intermediate/training/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_training_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-training/'+str(target_map)+'.geojson'

            path_to_log = 'Example_Output/'+str(target_map)+'.log'
            os.makedirs(os.path.dirname(path_to_log), exist_ok=True)
            os.makedirs(os.path.dirname(output_dir), exist_ok=True)
            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            class Logger(object):
                def __init__(self):
                    self.terminal = sys.stdout
                    self.log = open(path_to_log, 'a')
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

            
            flagging = link_for_integrated_processing.start_linking(target_map_name, input_image, None, path_to_intermediate, None, input_legend_segmentation, path_to_mapkurator_output, None, None, competition_custom, True, this_version)
            if flagging == True:
                link_for_postprocessing.start_linking_postprocessing(target_map, input_image, output_dir, path_to_intermediate, None, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom)
                print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count)+'   ...'+str(datetime.now()-runningtime_start))
                #legend_item_vector_evaluation_v02.single_evaluation(target_map_name, output_dir, groundtruth_dir, 'Example_Output', str(datetime.now()-runningtime_start))
            else:
                print('Disintegrity in map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
            #start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True, '1.2')
            this_map_count += 1

    



def main():
    this_batch_processing = str_to_bool(args.batch_processing)
    this_hackathon = str_to_bool(args.hackathon)
    if this_hackathon == True:
        batch_extract_main()
        return True
    if this_batch_processing == True:
        batch_main()
        return True

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.path_to_intermediate), exist_ok=True)

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


    input_image = args.input_image
    path_list = input_image.replace('\\','/').split('/')
    target_map_name_ext = path_list[-1]
    target_map_name = os.path.splitext(path_list[-1])[0]

    output_dir = args.output_dir
    path_to_intermediate = args.path_to_intermediate
    path_to_intermediate2 = os.path.join(args.path_to_intermediate, target_map_name)
    input_area_segmentation = args.input_area_segmentation
    input_legend_segmentation = args.input_legend_segmentation
    path_to_mapkurator_output = args.path_to_mapkurator_output
    preprocessing_for_cropping = str_to_bool(args.preprocessing_for_cropping)
    postprocessing_for_crs = str_to_bool(args.postprocessing_for_crs)
    competition_custom = str_to_bool(args.competition_custom)

    this_version = args.version

    
    input_groundtruth_dir = 'None'
    os.makedirs(os.path.dirname(path_to_intermediate2), exist_ok=True)


    flagging = link_for_integrated_processing.start_linking(target_map_name_ext, input_image, None, path_to_intermediate2, None, input_legend_segmentation, path_to_mapkurator_output, None, None, competition_custom, False, this_version)
    if flagging == True:
        link_for_postprocessing.start_linking_postprocessing(target_map_name, input_image, output_dir, path_to_intermediate2, None, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom)
    else:
        print('Disintegrity in map, please check the legend-area segmentation file. The targeted map may be missing from the json file if you have set "--competition_custom" to True...')
    return True
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image', type=str, default='OR_Carlton.tif')
    parser.add_argument('--output_dir', type=str, default='output_json')
    parser.add_argument('--preprocessing_for_cropping', type=str, default='True')
    parser.add_argument('--postprocessing_for_crs', type=str, default='True')
    parser.add_argument('--input_area_segmentation', type=str, default='OR_Carlton_area_extraction.tif')
    parser.add_argument('--input_legend_segmentation', type=str, default='')
    parser.add_argument('--path_to_intermediate', type=str, default='Solution_1130')
    parser.add_argument('--path_to_mapkurator_output', type=str, default='Legend/Map_Legend_Detection/ta1-feature-validation')

    parser.add_argument('--competition_custom', type=str, default='False')
    parser.add_argument('--batch_processing', type=str, default='False')
    parser.add_argument('--hackathon', type=str, default='False')

    parser.add_argument('--version', type=str, default='1.2')
    parser.add_argument('--log', type=str, default='log_file.txt')



    args = parser.parse_args()
    main()