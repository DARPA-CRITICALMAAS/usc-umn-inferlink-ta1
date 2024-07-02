import os
import argparse
import yaml
import time
import logging
import subprocess
from generate_inputs import GenerateInputs4LayoutLMv3
from generate_results import process_layoutlmv3_outputs
from human_input_process import HumanInputProcess

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def write_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def update_yaml(file_path, update_dict):
    data = read_yaml(file_path)
    for key, value in update_dict.items():
        data[key] = value
    write_yaml(data, file_path)

def execute_command(command, if_print_command):
    t1 = time.time()
    if if_print_command:
        print(command)
    try:
        subprocess.run(command, shell=True, check=True, capture_output = True) #stderr=subprocess.STDOUT)
        t2 = time.time()
        time_usage = t2 - t1 
        return {'time_usage':time_usage}
    except subprocess.CalledProcessError as err:
        error = err.stderr.decode('utf8')
        print('****', error)
        # format error message to one line
        error  = error.replace('\n','\t')
        error = error.replace(',',';')
        return {'error': error}
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_input_dir', type=str, required=True)
    parser.add_argument('--legend_input_dir', type=str, required=True)
    parser.add_argument('--map_name', type=str, required=True)
    parser.add_argument('--temp_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config_file_path', type=str, \
                        default='/ta1/inputs/modules/ptln_legend_item_description/cascade_layoutlmv3.yaml')
    parser.add_argument('--human_input_path', default='', type=str, help='Use human input')
    args = parser.parse_args()

    map_input_dir = args.map_input_dir
    legend_input_dir = args.legend_input_dir
    map_name = args.map_name
    temp_dir = args.temp_dir
    output_dir = args.output_dir
    config_file_path = args.config_file_path

    input_map_path = os.path.join(map_input_dir, map_name+'.tif')
    input_legend_path = os.path.join(legend_input_dir, map_name+'_map_segmentation.json')

    if human_input_path != '':
        human_input_process = HumanInputProcess(human_input_path, input_legend_path, input_map_path)       
        output_path = human_input_process.generate_module_output(output_dir)
    else:
        # step 1. prepare inputs for layoutmlv3
        layoutlmv3_generator = GenerateInputs4LayoutLMv3(input_map_path, input_legend_path, output_dir)
        layoutlmv3_input_dir = layoutlmv3_generator.generate_inputs4layoutlmv3()
        
        # step 2. update config and run layoutlmv3
        key_val_to_update = {'OUTPUT_DIR': output_dir, \
                            'PUBLAYNET_DATA_DIR_TEST': layoutlmv3_input_dir, \
                            # 'PUBLAYNET_DATA_DIR_TRAIN': layoutlmv3_input_dir, \
                            'CACHE_DIR': temp_dir}
        update_yaml(config_file_path, key_val_to_update)
        run_layoutlmv3_command = f"python examples/object_detection/train_net.py \
                                    --eval-only \
                                    --config-file {config_file_path} \
                                    --num-gpus 1"

        run_layoutlmv3_command  += ' 1> /dev/null'
        exe_ret = execute_command(run_layoutlmv3_command, True)
        if 'error' in exe_ret:
            logging.info(f'Error in running layoutlmv3 for {map_name}')
            print(f'=== Error in running layoutlmv3 for {map_name}')
            print(exe_ret)
        else:
            logging.info(f'layoutlmv3 ran successfully for {map_name}')
            print(f'=== layoutlmv3 ran successfully for {map_name}')
            print(exe_ret)

        #step 3. convert layoutlmv3 output to the required format for ptln modules
        layoutlmv3_output_json_path = os.path.join(output_dir, "inference/coco_instances_results.json")
        layoutlmv3_input_json_path = layoutlmv3_input_dir+'.json'
        process_layoutlmv3_outputs(layoutlmv3_output_json_path, layoutlmv3_input_dir, \
                                layoutlmv3_input_json_path, map_name, output_dir)
    return 0

if __name__ == "__main__":
    main()