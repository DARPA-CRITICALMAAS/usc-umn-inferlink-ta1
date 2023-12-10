
import os
import map_area_segmenter
import area_json_generator

def main():
    if not os.path.exists(args.intermediate_dir):
        os.makedirs(args.intermediate_dir)

    map_area_segmenter.cropping_worker(args.input_path, args.binary_output_path, args.intermediate_dir)
    area_json_generator.json_generator(args.binary_output_path, args.json_output_path, args.intermediate_dir)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='image.tif')
    parser.add_argument('--binary_output_path', type=str, default='image_seg.tif')
    parser.add_argument('--json_output_path', type=str, default='image_seg.json')
    parser.add_argument('--intermediate_dir', type=str, default='Intermediate')

    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()
