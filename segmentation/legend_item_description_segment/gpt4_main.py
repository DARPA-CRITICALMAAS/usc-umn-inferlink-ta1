from argparse import ArgumentParser
from gpt4_input_generation import main as generate_gpt4_input
from symbol_description_extraction_gpt4 import main as extract_symbol_description
from postprocess import combine_json_files_from_gpt, combine_json_files_gpt_and_symbol_legend
import os

parser = ArgumentParser()
parser.add_argument('--map_dir',
                    type=str,
                   default='/data/weiweidu/criticalmaas_data/training')
parser.add_argument('--legend_json_path',
                   type=str,
                   default=None)
parser.add_argument('--symbol_json_dir',
                   type=str,
                   default=None)
parser.add_argument('--map_name',
                   type=str,
                   default=None)
parser.add_argument('--gpt4_input_dir',
                  type=str,
                  default=None)
parser.add_argument('--gpt4_intermediate_dir',
                   type=str,
                   default=None)
parser.add_argument('--gpt4_output_dir',
                   type=str,
                   default=None)

if __name__ == '__main__':
    args = parser.parse_args()
    all_sym_bbox = generate_gpt4_input(args.map_dir, args.legend_json_path, args.symbol_json_dir,\
                       args.map_name, args.gpt4_input_dir)
    
    import os
    for root, dirs, files in os.walk(args.gpt4_input_dir):
        for f_name in files:
            if args.map_name not in f_name: 
                continue
            print(f'*** processing {f_name} ***')
            image_name = f_name.split('.')[0]
            extract_symbol_description(image_name, args.map_dir, all_sym_bbox, args.gpt4_input_dir, args.gpt4_intermediate_dir)
    
    # combine the gpt results into one json file
    output_path = os.path.join(args.gpt4_intermediate_dir, args.map_name+'.json')
    combine_json_files_from_gpt(args.gpt4_intermediate_dir, args.map_name, output_path)
    
    output_path = os.path.join(args.gpt4_output_dir, args.map_name+'.json')
    combine_json_files_gpt_and_symbol_legend(args.symbol_json_dir, args.legend_json_path, args.gpt4_intermediate_dir, args.map_name, output_path)