
import os
import argparse
import link_inference
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
    output_dir = args.output_dir
    path_to_intermediate = args.path_to_intermediate
    input_area_segmentation = args.input_area_segmentation
    input_legend_segmentation = args.input_legend_segmentation
    path_to_mapkurator_output = args.path_to_mapkurator_output
    preprocessing_for_cropping = str_to_bool(args.preprocessing_for_cropping)
    postprocessing_for_crs = str_to_bool(args.postprocessing_for_crs)

    path_list = input_image.replace('\\','/').split('/')
    target_map_name = path_list[-1]
    #target_map_name = os.path.splitext(path_list[-1])[0]

    link_inference.start_linking(target_map_name, input_image, output_dir, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, preprocessing_for_cropping, postprocessing_for_crs)



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

    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--log', type=str, default='log_file.txt')

    args = parser.parse_args()
    main()