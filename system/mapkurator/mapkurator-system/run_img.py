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

def execute_command(command, if_print_command):
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
    geotiff_output_dir = os.path.join(output_folder, expt_name,  'geotiff')
    cropping_output_dir = os.path.join(output_folder, expt_name, 'crop/')
    spotting_output_dir = os.path.join(output_folder, expt_name,  'spotter/' + spotter_expt_name)
    stitch_output_dir = os.path.join(output_folder, expt_name, 'stitch/' + spotter_expt_name)
    postocr_output_dir = os.path.join(output_folder, expt_name, 'postocr/'+ spotter_expt_name)
    geojson_output_dir = os.path.join(output_folder, expt_name, 'geojson_' + spotter_expt_name + '/')

    
    if not os.path.isdir(expt_out_dir):
        os.makedirs(expt_out_dir)

    # ------------------------ Get image dimension  ------------------------------
    if module_get_dimension:
        for index, record in sample_map_df.iterrows():
            external_id = record.external_id

            img_path = sample_map_df['external_id'].iloc[index]
            map_name = os.path.basename(img_path).split('.')[0]
            width, height = get_img_dimension(img_path)          
            
    # ------------------------- Generate geotiff ------------------------------
    time_start =  time.time()
    if module_gen_geotiff:
        os.chdir(os.path.join(map_kurator_system_dir ,'m1_geotiff'))
        
        if not os.path.isdir(geotiff_output_dir):
            os.makedirs(geotiff_output_dir)

        # use converted jpg folder instead of original sid folder
        run_geotiff_command = 'python convert_image_to_geotiff.py --sid_root_dir /data2/rumsey_sid_to_jpg/ --input_dir_path '+ input_img_path +' --out_geotiff_dir '+geotiff_output_dir  # can change params in argparse
        exe_ret = execute_command(run_geotiff_command, if_print_command)
        if 'error' in exe_ret:
            error = exe_ret['error']
        elif 'time_usage' in exe_ret:
            time_usage = exe_ret['time_usage']

    time_geotiff = time.time()
    

    # ------------------------- Image cropping  ------------------------------
    if module_cropping:
        try:
            if len(os.listdir(input_img_path))==0:
                raise Exception("Empty input directory found. Please ensure there is a file in the input folder.")
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
                execute_command(f'python setup.py build develop 1> /dev/null',if_print_command)
                
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
        os.chdir(os.path.join(map_kurator_system_dir ,'m3_image_geojson'))
        
        if not os.path.isdir(stitch_output_dir):
            os.makedirs(stitch_output_dir)

        for index, record in sample_map_df.iterrows():
            img_path = sample_map_df['external_id'].iloc[index]
            map_name = os.path.basename(img_path).split('.')[0]

            stitch_input_dir = os.path.join(spotting_output_dir, map_name)
            output_geojson = os.path.join(stitch_output_dir, map_name + '.geojson')

            run_stitch_command = 'python stitch_output.py --input_dir '+stitch_input_dir + ' --output_geojson ' + output_geojson
            
            
            exe_ret = execute_command(run_stitch_command, if_print_command)
            
            if 'error' in exe_ret:
                print(exe_ret['error'])

            
    time_img_geojson = time.time()

    # ------------------------- post-OCR ------------------------------
    if module_post_ocr:
        
        os.chdir(os.path.join(map_kurator_system_dir, 'm6_post_ocr'))

        if not os.path.isdir(postocr_output_dir):
            os.makedirs(postocr_output_dir)

        for index, record in sample_map_df.iterrows():
            
            external_id = record.external_id
            img_path = sample_map_df['external_id'].iloc[index]
            map_name = os.path.basename(img_path).split('.')[0]

            input_geojson_file = os.path.join(stitch_output_dir, map_name + '.geojson')
            geojson_postocr_output_file = os.path.join(postocr_output_dir, map_name + '.geojson')
            print('input_geojson_file',input_geojson_file)
            print('geojson_postocr_output_file',geojson_postocr_output_file)
            run_postocr_command = 'python lexical_search.py --in_geojson_dir '+ input_geojson_file +' --out_geojson_dir '+ geojson_postocr_output_file

            exe_ret = execute_command(run_postocr_command, if_print_command)
            print('exe_ret',exe_ret)

    time_post_ocr = time.time()
    
    
     # ------------------------- Convert image coordinates to geocoordinates ------------------------------
    if module_geocoord_geojson:
        os.chdir(os.path.join(map_kurator_system_dir, 'm4_geocoordinate_converter'))
        
        if not os.path.isdir(geojson_output_dir):
            os.makedirs(geojson_output_dir)

        for index, record in sample_map_df.iterrows():
            external_id = record.external_id
            in_geojson = os.path.join(output_folder, postocr_output_dir+'/') + external_id.strip("'").replace('.', '') + ".geojson"

            run_converter_command = 'python convert_geojson_to_geocoord.py --input_dir_path '+ os.path.join(map_kurator_system_dir, input_img_path) +' --in_geojson_file '+ in_geojson +' --out_geojson_dir '+ os.path.join(map_kurator_system_dir, geojson_output_dir)
            exe_ret = execute_command(run_converter_command, if_print_command)

    time_geocoord_geojson = time.time()

    # ------------------------- Link entities in OSM ------------------------------
    if module_entity_linking:
        os.chdir(os.path.join(map_kurator_system_dir, 'm5_entity_linker'))
        
        geojson_linked_output_dir = os.path.join(map_kurator_system_dir, 'm5_entity_linker', 'data/100_maps_geojson_abc_linked/')
        if not os.path.isdir(geojson_output_dir):
            os.makedirs(geojson_output_dir)

        run_linker_command = 'python entity_linker.py --input_dir_path '+ input_img_path +' --in_geojson_dir '+ geojson_output_dir +' --out_geojson_dir '+ geojson_linked_output_dir
        execute_command(run_linker_command, if_print_command)

    time_entity_linking = time.time()

    # --------------------- Time usage logging --------------------------
    print('\n')
    logging.info('Time for generating geotiff: %d', time_geotiff - time_start)
    logging.info('Time for Cropping : %d',time_cropping - time_geotiff)
    logging.info('Time for text spotting : %d',time_text_spotting - time_cropping)
    logging.info('Time for generating geojson in img coordinate : %d',time_img_geojson - time_text_spotting)
    logging.info('Time for generating geojson in geo coordinate : %d',time_geocoord_geojson - time_img_geojson)
    logging.info('Time for entity linking : %d',time_entity_linking - time_geocoord_geojson)
    logging.info('Time for post OCR : %d',time_post_ocr - time_img_geojson)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--map_kurator_system_dir', type=str, default='/home/maplord/rumsey/mapkurator-system/')
    parser.add_argument('--text_spotting_model_dir', type=str, default='/home/maplord/rumsey/TESTR/')
    parser.add_argument('--input_dir_path', type=str, default='m1_geotiff/data/sample_US_jp2_100_maps.csv') # Original: sample_US_jp2_100_maps.csv
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

    
