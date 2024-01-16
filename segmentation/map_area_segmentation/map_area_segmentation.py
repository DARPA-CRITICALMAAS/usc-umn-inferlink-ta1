
import os
import map_area_segmenter
import area_json_generator

import sys

def main():
    #if not os.path.exists(args.intermediate_dir):
        #os.makedirs(args.intermediate_dir)
    os.makedirs(os.path.dirname(args.intermediate_dir), exist_ok=True)
    os.makedirs(os.path.dirname(args.json_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(args.log, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger()

    map_area_segmenter.cropping_worker(args.input_path, args.binary_output_path, args.intermediate_dir)
    area_json_generator.json_generator(args.binary_output_path, args.json_output_path, args.intermediate_dir)

    exit(0)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='image.tif')
    parser.add_argument('--binary_output_path', type=str, default='image_seg.tif')
    parser.add_argument('--json_output_path', type=str, default='image_seg.json')
    parser.add_argument('--intermediate_dir', type=str, default='Intermediate')
    parser.add_argument('--version', type=str, default='0')
    parser.add_argument('--log', type=str, default='log_file.txt')

    args = parser.parse_args()
    
    #print(f"Processing output for: {args.result_name}")
    main()
