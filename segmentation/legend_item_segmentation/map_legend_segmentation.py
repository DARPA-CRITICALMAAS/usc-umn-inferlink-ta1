
import os
import argparse
import link_for_metadata_preprocessing
import link_for_metadata_postprocessing
import link_for_loam_inference
import link_for_postprocessing
#import map_area_segmenter

import sys


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
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

    
    input_groundtruth_dir = 'None'
    os.makedirs(os.path.dirname(path_to_intermediate2), exist_ok=True)

    
    target_dir_img = 'LOAM_Intermediate/data/cma/imgs'
    target_dir_mask = 'LOAM_Intermediate/data/cma/masks'

    target_dir_img_small = 'LOAM_Intermediate/data/cma_small/imgs'
    target_dir_mask_small = 'LOAM_Intermediate/data/cma_small/masks'

    link_for_metadata_preprocessing.start_linking(target_map_name_ext, input_image, None, path_to_intermediate2, None, input_legend_segmentation, path_to_mapkurator_output, None, None, competition_custom)
    link_for_metadata_postprocessing.start_data_postprocessing(target_map_name, path_to_intermediate, input_groundtruth_dir, target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small, 1024, False)
    link_for_loam_inference.loam_inference(
            input_filtering_new_dataset = False,
            input_filtering_threshold = 0.02,
            input_k_fold_testing = 1,
            input_crop_size = 1024,
            input_separate_validating_set = False,
            input_reading_predefined_testing = True,
            input_training_needed = False,
            input_targeted_map_file = 'targeted_map.csv',
            input_path_to_tif = input_image,
            input_groundtruth_dir = 'None',
            input_performance_evaluation = False
        )
    
    link_for_postprocessing.start_linking_postprocessing(target_map_name, input_image, output_dir, path_to_intermediate2, None, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom)
    
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

    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--log', type=str, default='log_file.txt')

    args = parser.parse_args()
    main()