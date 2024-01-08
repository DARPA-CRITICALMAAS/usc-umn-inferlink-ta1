import os
import subprocess
import glob
import argparse
import time
import logging
import pandas as pd
import pdb
import datetime
from PIL import Image 
from utils import get_img_path_from_external_id, get_img_path_from_external_id_and_image_no

#this code is the case for getting an input as folders which include images.  
#tested image : /home/maplord/rumsey/mapkurator-system/data/100_maps_crop/crop_leeje_2/test_run_img/

logging.basicConfig(level=logging.INFO)
Image.MAX_IMAGE_PIXELS=None # allow reading huge images

def execute_command(command, if_print_command=False):
    t1 = time.time()

    if if_print_command:
        print(command)

    try:
        subprocess.run(command, shell=True,check=True, capture_output = True) #stderr=subprocess.STDOUT)
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
    time_start = time.time()
    map_kurator_system_dir = os.path.join(os.getcwd())
    # -------------------------  Pass arguments -----------------------------------------
    text_spotting_model_dir = args.text_spotting_model_dir
    sample_map_path = map_kurator_system_dir+args.sample_map_path
    output_folder = map_kurator_system_dir+args.output_folder
    spotter_model = args.spotter_model
    spotter_config = args.spotter_config

    gpu_id = args.gpu_id
    expt_name = args.expt_name
    spotter_expt_name = args.spotter_expt_name
    # module_get_dimension = True
    # module_gen_geotiff = args.module_gen_geotiff
    module_cropping = True
    module_text_spotting = True
    module_img_geojson = True
    # module_geocoord_geojson = args.module_geocoord_geojson 
    # module_entity_linking = args.module_entity_linking
    # module_post_ocr = args.module_post_ocr

    if_print_command = args.print_command

    input_img_path = sample_map_path 
    sample_map_df = pd.DataFrame(columns = ["external_id"])
    for images in os.listdir(input_img_path):
        tmp_path={"external_id": input_img_path+images}
        sample_map_df=sample_map_df.append(tmp_path,ignore_index=True)

    expt_out_dir = os.path.join(output_folder, expt_name)
    geotiff_output_dir = os.path.join(output_folder, expt_name,  'geotiff')
    cropping_output_dir = os.path.join(output_folder, expt_name, 'crop/')
    spotting_output_dir = os.path.join(output_folder, expt_name,  'spotter/' + spotter_expt_name)
    stitch_output_dir = os.path.join(output_folder, expt_name, 'stitch/' + spotter_expt_name)
    if not os.path.isdir(expt_out_dir):
        os.makedirs(expt_out_dir)        

    # ------------------------- Image cropping  ------------------------------
    if module_cropping:
        try:
            if len(os.listdir(input_img_path))==0:
                raise Exception("Empty input directory found. Please ensure there is a file in ../data/test_imgs/sample_input/")
            else:
                for index, record in sample_map_df.iterrows():
                    external_id = record.external_id
                    img_path = sample_map_df['external_id'].iloc[index]
                    map_name = os.path.basename(img_path).split('.')[0]

                    os.chdir(os.path.join(map_kurator_system_dir ,'m2_detection_recognition'))
                    if not os.path.isdir(cropping_output_dir):
                        os.makedirs(cropping_output_dir)
                    
                    run_crop_command = 'python crop_img.py --img_path '+img_path + ' --output_dir '+ cropping_output_dir
                    exe_ret = execute_command(run_crop_command, if_print_command)
                    if 'error' in exe_ret:
                        print(exe_ret['error'])
                    elif 'time_usage' in exe_ret:
                        time_usage = exe_ret['time_usage']
        except Exception as e:
            logging.info(e)

    time_cropping = time.time()
                
    # ------------------------- Text Spotting (patch level) ------------------------------
    if module_text_spotting:
        try:
            if len(os.listdir(cropping_output_dir))==0:
                raise Exception("Empty input directory found. Please ensure there is at least one cropped image in cropping directory ../data/test_imgs/sample_output/crop/...")
            else:           
                assert os.path.exists(spotter_config), "Config file for spotter must exist!"
                os.chdir(text_spotting_model_dir) 
                execute_command(f'python setup.py build develop 1> /dev/null')
                
                for index, record in sample_map_df.iterrows():

                    external_id = record.external_id

                    img_path = sample_map_df['external_id'].iloc[index]
                    map_name = os.path.basename(img_path).split('.')[0]

                    map_spotting_output_dir = os.path.join(spotting_output_dir, map_name)
                    if not os.path.isdir(map_spotting_output_dir):
                        os.makedirs(map_spotting_output_dir)
                
                    print(os.path.join(cropping_output_dir,map_name))
                    if spotter_model in ['testr', 'spotter_v2']:
                        run_spotting_command = f'CUDA_VISIBLE_DEVICES={gpu_id} python tools/inference.py --config-file {spotter_config} --output_json --input {os.path.join(cropping_output_dir,map_name)} --output {map_spotting_output_dir}'
                    else:
                        raise NotImplementedError
                    
                    run_spotting_command  += ' 1> /dev/null'
                    exe_ret = execute_command(run_spotting_command, if_print_command)
                    if 'error' in exe_ret:
                        print(exe_ret['error'])
                    logging.info('Done text spotting for %s', map_name)
                    
        except Exception as e:
            logging.info(e)

    time_text_spotting = time.time()
    

    # ------------------------- Image coord geojson (map level) ------------------------------
    if module_img_geojson:
        try:
            if len(os.listdir(spotting_output_dir))==0:
                raise Exception("Empty input directory found. Please ensure there is at least one image in spotting directory ../data/test_imgs/sample_output/spotter/...")
            else:
                os.chdir(os.path.join(map_kurator_system_dir ,'m3_image_geojson'))
                
                if not os.path.isdir(stitch_output_dir):
                    os.makedirs(stitch_output_dir)

                for index, record in sample_map_df.iterrows():
                    img_path = sample_map_df['external_id'].iloc[index]
                    map_name = os.path.basename(img_path).split('.')[0]

                    stitch_input_dir = os.path.join(spotting_output_dir, map_name)
                    output_geojson = os.path.join(stitch_output_dir, map_name + '.geojson')

                    run_stitch_command = 'python stitch_output.py --input_dir '+stitch_input_dir + ' --output_geojson ' + output_geojson + ' --eval_only'
                                
                    exe_ret = execute_command(run_stitch_command, if_print_command)
                    
                    if 'error' in exe_ret:
                        print(exe_ret['error'])
        except Exception as e:
            logging.info(e)

    time_img_geojson = time.time()

    # --------------------- Time usage logging --------------------------
    print('\n')
    logging.info('Time for Cropping : %.4f',time_cropping-time_start)
    logging.info('Time for text spotting : %.4f',time_text_spotting - time_cropping)
    logging.info('Time for generating geojson in img coordinate : %.4f',time_img_geojson - time_text_spotting)

def main():
    os.chdir('/home/mapkurator-system/') #/home/mapkurator-system/
    print("Working in directory: ", os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument('--text_spotting_model_dir', type=str, default='/home/spotter_v2/PALEJUN/')
    parser.add_argument('--sample_map_path', type=str, default='/data/test_imgs/sample_input') # Original: sample_US_jp2_100_maps.csv
    parser.add_argument('--output_folder', type=str, default='/data/test_imgs/sample_output') # Original: /data2/rumsey_output
    parser.add_argument('--spotter_model', type=str, default='spotter_v2', choices=['testr', 'spotter_v2'], 
        help='Select text spotting model option from ["testr", "spotter_v2"]') # select text spotting model
    parser.add_argument('--spotter_config', type=str, default='/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml',
        help='Path to the config file for text spotting model')
    parser.add_argument('--print_command', default=False, action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    
    # the arguments below are set to enable unit testing of this file.
    parser.add_argument('--spotter_expt_name', type=str, default='intermediate_results',
        help='Name of spotter experiment, if empty using config file name') 
    parser.add_argument('--expt_name', type=str, default='intermediate_results') 
                    
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    run_pipeline(args)

if __name__ == '__main__':

    main()

    
