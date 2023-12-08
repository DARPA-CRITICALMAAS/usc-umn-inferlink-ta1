
import os
import argparse
import link_inference
#import map_area_segmenter


def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def main():
    input_image = args.input_image
    output_dir = args.output_dir
    path_to_intermediate = args.path_to_intermediate
    input_area_segmentation = args.input_area_segmentation
    input_legend_segmentation = args.input_legend_segmentation
    path_to_mapkurator_output = args.path_to_mapkurator_output
    preprocessing_for_cropping = str_to_bool(args.preprocessing_for_cropping)
    postprocessing_for_crs = str_to_bool(args.postprocessing_for_crs)

    link_inference.start_linking(input_image, output_dir, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, preprocessing_for_cropping, postprocessing_for_crs)



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

    args = parser.parse_args()
    main()