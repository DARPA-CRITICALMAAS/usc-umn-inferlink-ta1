import os
import subprocess
import glob
import argparse
import time
import logging
import pandas as pd
import pdb
import yaml
import datetime
from PIL import Image 
from utils import get_img_path_from_external_id, get_img_path_from_external_id_and_image_no

#this code is the case for getting an input as folders which include images.  
#tested image : /home/maplord/rumsey/mapkurator-system/data/100_maps_crop/crop_leeje_2/test_run_img/

logging.basicConfig(level=logging.INFO)
Image.MAX_IMAGE_PIXELS=None # allow reading huge images

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def write_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
    
def update_yaml(file_path, key, value):
    data = read_yaml(file_path)
    keys = key.split('.')
    data['MODEL']['WEIGHTS'] = value
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
        # format error message to one line
        error  = error.replace('\n','\t')
        error = error.replace(',',';')
        return {'error': error}


def get_img_dimension(img_path):
    map_img = Image.open(img_path) 
    width, height = map_img.size 

    return width, height


def run_pipeline(args):
    # -------------------------  Pass arguments -----------------------------------------
    map_kurator_system_dir = args.map_kurator_system_dir
    text_spotting_model_dir = args.text_spotting_model_dir
    input_dir_path = args.input_dir_path
    expt_name = args.expt_name
    output_folder = args.output_folder

    module_get_dimension = args.module_get_dimension
    module_gen_geotiff = args.module_gen_geotiff
    module_cropping = args.module_cropping
    module_text_spotting = args.module_text_spotting
    module_img_geojson = args.module_img_geojson 
    module_geocoord_geojson = args.module_geocoord_geojson 
    module_entity_linking = args.module_entity_linking
    module_post_ocr = args.module_post_ocr

    spotter_model = args.spotter_model
    spotter_config = args.spotter_config
    spotter_expt_name = args.spotter_expt_name
    gpu_id = args.gpu_id
    
    if_print_command = args.print_command

    input_img_path = input_dir_path 
    sample_map_df = pd.DataFrame(columns = ["external_id"])
    for images in os.listdir(input_img_path):
            tmp_path={"external_id": input_img_path+images}
            sample_map_df=sample_map_df.append(tmp_path,ignore_index=True)

    expt_out_dir = os.path.join(output_folder, expt_name)
    spotting_output_dir = os.path.join(output_folder, expt_name,  'spotter/' + spotter_expt_name)
    stitch_output_dir = os.path.join(output_folder, expt_name, 'stitch/' + spotter_expt_name)


    if not os.path.isdir(expt_out_dir):
        os.makedirs(expt_out_dir)
    
    # ------------------------- Text Spotting (patch level) ------------------------------
    if module_text_spotting:
        start_time = time.time()
        try:
            if False:#len(os.listdir(cropping_output_dir))==0:
                raise Exception("Empty input directory found. Please ensure there is at least one cropped image in cropping directory ../data/test_imgs/sample_output/crop/...")
            else:           
                assert os.path.exists(spotter_config), "Config file for spotter must exist!"
                os.chdir(text_spotting_model_dir) 
                execute_command(f'python setup.py build develop 1> /dev/null',if_print_command)
                
#                 for index, record in sample_map_df.iterrows():

#                     external_id = record.external_id

#                     img_path = sample_map_df['external_id'].iloc[index]
#                     map_name = os.path.basename(img_path).split('.')[0]

                map_spotting_output_dir = os.path.join(spotting_output_dir)#, map_name)
                if not os.path.isdir(map_spotting_output_dir):
                    os.makedirs(map_spotting_output_dir)

                if spotter_model in ['testr', 'spotter_v2']:
                    update_yaml(spotter_config, 'MODEL.WEIGHTS', args.model_weight_path)
                    run_spotting_command = f'CUDA_VISIBLE_DEVICES={gpu_id} python tools/inference.py --config-file {spotter_config} --output_json --input {os.path.join(args.input_dir_path)} --output {map_spotting_output_dir}'

                else:
                    raise NotImplementedError

                run_spotting_command  += ' 1> /dev/null'
                exe_ret = execute_command(run_spotting_command, if_print_command)
                if 'error' in exe_ret:
                    print(exe_ret['error'])
                logging.info('Done text spotting for %s', args.input_dir_path)
                    
        except Exception as e:
            logging.info(e)

    time_text_spotting = time.time()
    

    # ------------------------- Image coord geojson (map level) ------------------------------
    if module_img_geojson:
        os.chdir(os.path.join(map_kurator_system_dir ,'m3_image_geojson'))
        
        if not os.path.isdir(stitch_output_dir):
            os.makedirs(stitch_output_dir)

        map_name = '_'.join(args.input_dir_path.split('/')[-1].split('_')[:-2])

        stitch_input_dir = spotting_output_dir
        output_geojson = os.path.join(stitch_output_dir, map_name + '.geojson')

        run_stitch_command = 'python stitch_output.py --input_dir '+stitch_input_dir + ' --output_geojson ' + output_geojson


        exe_ret = execute_command(run_stitch_command, True)

        if 'error' in exe_ret:
                print(exe_ret['error'])

            
    time_img_geojson = time.time()


    # --------------------- Time usage logging --------------------------
    print('\n')
    logging.info('Time for text spotting : %d',time_text_spotting - start_time)
    logging.info('Time for generating geojson in img coordinate : %d',time_img_geojson - time_text_spotting)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_kurator_system_dir', type=str, default='/home/maplord/rumsey/mapkurator-system/')
    parser.add_argument('--text_spotting_model_dir', type=str, default='/home/maplord/rumsey/TESTR/')
    parser.add_argument('--input_dir_path', type=str, default='m1_geotiff/data/sample_US_jp2_100_maps.csv') 
    parser.add_argument('--model_weight_path', type=str, default='../spotter_v2/PALEJUN/weights/synthmap_pretrain/model_final.pth')
    parser.add_argument('--output_folder', type=str, default='/data2/rumsey_output') # Original: /data2/rumsey_output
    parser.add_argument('--expt_name', type=str, default='1000_maps') # output prefix 
    
    parser.add_argument('--module_get_dimension', default=False, action='store_true')
    parser.add_argument('--module_gen_geotiff', default=False, action='store_true')
    parser.add_argument('--module_cropping', default=False, action='store_true')
    parser.add_argument('--module_text_spotting', default=False, action='store_true')
    parser.add_argument('--module_img_geojson', default=False, action='store_true')
    parser.add_argument('--module_geocoord_geojson', default=False, action='store_true')
    parser.add_argument('--module_entity_linking', default=False, action='store_true')
    parser.add_argument('--module_post_ocr', default=False, action='store_true')

    parser.add_argument('--spotter_model', type=str, default='spotter_v2', choices=['abcnet', 'testr', 'spotter_v2','spotter_v3'], 
        help='Select text spotting model option from ["abcnet","testr", "testr_v2"]') # select text spotting model
    parser.add_argument('--spotter_config', type=str, default='/home/maplord/rumsey/TESTR/configs/TESTR/SynMap/SynMap_Polygon.yaml',
        help='Path to the config file for text spotting model')
    parser.add_argument('--spotter_expt_name', type=str, default='testr_syn',
        help='Name of spotter experiment, if empty using config file name') 

    parser.add_argument('--print_command', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    run_pipeline(args)

if __name__ == '__main__':

    main()

    
