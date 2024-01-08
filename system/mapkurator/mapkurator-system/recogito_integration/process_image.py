import argparse
import geojson
import glob
import os
import uuid

from wmts_handler import WMTSHandler
from image_handler import ImageHandler
from iiif_handler import IIIFHandler

import cv2
import numpy as np
import json
from shapely.geometry import Polygon

os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import sys
import tensorflow as tf

import time
import subprocess
import logging

print(tf.__file__)
print(tf.__version__)

# helper function to execute linux commands.
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

def write_annotation(input_dir, output_dir,img_id,img_path,handler=None):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        
    # read mapkurator-system stitch module output.
    file_list = glob.glob(input_dir + '/*.geojson')
    file_list = sorted(file_list)
    if len(file_list) == 0:
        logging.warning('No files found for %s' % input_dir)
    
    # run through all files available in the folder.
    for input_file in file_list:
        with open(input_file) as img_geojson:
            img_data = geojson.load(img_geojson)

            if handler is not None: 
                # perform this operation for WMTS tiles only
                # based on the tile info, convert from image coordinate system to EPSG：4326
                # assumes that the tilesize = 256x256

                tile_info = handler.tile_info
                min_tile_x = tile_info['min_x']
                min_tile_y = tile_info['min_y']
                
                for img_feature in img_data['features']: 
                    latlon_poly_list = []
                    polygon = img_feature['geometry']['coordinates'][0]

                    if np.array(polygon).shape[0] == 0:
                        continue   
                    # process each polygon 
                    poly_x_list , poly_y_list = np.array(polygon)[:,0], np.array(polygon)[:,1] 

                    # get corresponding tile index in the current map, i.e. tile shift range from min_tile_x ,min_tile_y
                    temp_tile_x_list, temp_tile_y_list = np.floor(poly_x_list/ 256.),  np.floor(poly_y_list/256.)

                    # compute the starting tile idx that the polygon point lies in
                    tile_x_list, tile_y_list = min_tile_x + temp_tile_x_list , min_tile_y + temp_tile_y_list

                    # get polygon point pixel location in its current tile
                    remainder_x_list, remainder_y_list = poly_x_list/256. - temp_tile_x_list , poly_y_list/256. - temp_tile_y_list

                    # final position in EPSG:3857? 
                    tile_x_list, tile_y_list = tile_x_list + remainder_x_list, tile_y_list + remainder_y_list  

                    # convert to EPSG:4326
                    lat_list, lon_list = handler._tile2latlon_list(tile_x_list, tile_y_list)

                    # x=long, y = lat. so need to flip 
                    # latlon_poly = [[x,y] for x,y in zip(lon_list, lat_list)]
                    latlon_poly = [["{:.6f}".format(x),"{:.6f}".format(y)] for x,y in zip(lon_list, lat_list)]

                    # latlon_poly_list.append(latlon_poly)
                    # polygon=latlon_poly
                    img_feature['geometry']['coordinates'][0]=latlon_poly


            # Generate web annotations: https://www.w3.org/TR/annotation-model/
            annotations = []
            for img_feature in img_data['features']: 
                polygon = img_feature['geometry']['coordinates'][0]
                svg_polygon_coords = ' '.join([f"{x},{y}" for x, y in polygon])
                annotation = {                
                    "@context": "http://www.w3.org/ns/anno.jsonld",
                    "id": "",
                    "type": "Annotation",
                    "body": [{
                        "type": "TextualBody",
                        "purpose": "transcribing",
                        "value": img_feature['properties']['text']
                        },
                        {
                        "type": "Dataset",
                        "format": "application/json",
                        "purpose": "documenting",
                        "value": {}
                        }],    
                    "target": {
                        "selector": [{
                        "type": "SvgSelector",
                        "value": f"<svg><polygon points='{svg_polygon_coords}'></polygon></svg>"
                        }]
                    }
                }
                annotations.append(annotation)
        
        annotation_file = os.path.join(output_dir, img_id+'_annotations.json')
         
        # save annotations file 
        with open(annotation_file, "w") as data_file:
            json.dump(annotations, data_file, indent=2)
                
        execute_command('rm -rf '+ img_path+'*',True)
        execute_command('rm -rf '+ output_dir+'intermediate_results',True)

if __name__ == "__main__":
    os.chdir('/home/mapkurator-system/') 
    print("Working in directory: ", os.getcwd())

    parser = argparse.ArgumentParser()

    arg_parser_common = argparse.ArgumentParser(add_help=False)
    arg_parser_common.add_argument('--dst', required=True, type=str, help='path to output annotations file')
    arg_parser_common.add_argument('--filename', required=False, type=str, help='output filename prefix')
    arg_parser_common.add_argument('--coord', default = 'img_coord', required=False, type=str, choices = ['img_coord' ,'epsg4326'], help='return annotation in image coord or EPSG:4326')
    arg_parser_common.add_argument('--print_command', default=False, action='store_true')
    arg_parser_common.add_argument('--spotter_model', type=str, default='spotter_v2', choices=['testr', 'spotter_v2'], 
        help='Select text spotting model option from ["testr", "spotter_v2"]') # select text spotting model
    arg_parser_common.add_argument('--spotter_config', type=str, default='/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml',
        help='Path to the config file for text spotting model')
    arg_parser_common.add_argument('--text_spotting_model_dir', type=str, default='/home/spotter_v2/PALEJUN/')
    arg_parser_common.add_argument('--gpu_id', type=int, default=0)

    subparsers = parser.add_subparsers(dest='subcommand')

    arg_parser_wmts = subparsers.add_parser('wmts', parents=[arg_parser_common],
                                            help='generate annotations for wmts input type')
    arg_parser_wmts.add_argument('--url', required=True, type=str, help='getCapabilities url')
    arg_parser_wmts.add_argument('--boundary', required=True, type=str, help='desired region boundary in GeoJSON')
    arg_parser_wmts.add_argument('--zoom', default=14, type=int, help='desired zoom level')

    arg_parser_iiif = subparsers.add_parser('iiif', parents=[arg_parser_common],
                                            help='generate annotations for iiif input type')
    arg_parser_iiif.add_argument('--url', required=True, type=str, help='IIIF manifest url')

    arg_parser_raw_input = subparsers.add_parser('file', parents=[arg_parser_common])
    arg_parser_raw_input.add_argument('--src', required=True, type=str, help='path to input image')

    args = parser.parse_args()

    map_path = None
    output_dir = args.dst
    gpu_id=args.gpu_id
    spotter_model = args.spotter_model
    spotter_config = args.spotter_config
    spotter_dir = args.text_spotting_model_dir

    if args.filename is not None:
        img_id = args.filename
    else:
        img_id = str(uuid.uuid4())

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if args.coord == 'epsg4326':
        assert args.subcommand == 'wmts'

    if args.subcommand == 'wmts':
        start_download = time.time()
        src = os.getcwd()+"/data/test_imgs/sample_input/" # set the destination folder for storing mapkurator-system input
        wmts_handler = WMTSHandler(url=args.url, bounds=args.boundary, zoom=args.zoom, output_dir=src, img_filename=img_id + '_stitched.jpg')
        wmts_handler.process_wmts() 
        
        end_download = time.time()
        
        if_print_command = args.print_command
        run_img_command = "python /home/mapkurator-system/run_pipeline.py "\
        "--output_folder=/data/test_imgs/sample_output/ "\
        "--text_spotting_model_dir="+spotter_dir+" "\
        "--sample_map_path=/data/test_imgs/sample_input/ "\
        "--spotter_model="+spotter_model+" "\
        "--spotter_config="+spotter_config+" "\
        "--gpu_id="+str(gpu_id)

        exe_ret = execute_command(run_img_command, if_print_command)
        if 'error' in exe_ret:
            error = exe_ret['error']
        elif 'time_usage' in exe_ret:
            time_usage = exe_ret['time_usage']

        # read from output of stitch module and write to output folder
        input_dir =os.getcwd()+"/data/test_imgs/sample_output/intermediate_results/stitch/intermediate_results/"

        if args.coord == 'img_coord':
            annotation_file = write_annotation(input_dir, args.dst,args.filename,"data/test_imgs/sample_input/")
        else:
            annotation_file = write_annotation(input_dir, args.dst,args.filename,"data/test_imgs/sample_input/",handler=wmts_handler)
        
        end_detection = time.time()

        print('download time: ', end_download - start_download)
        print('detection time: ', end_detection - end_download)

    if args.subcommand == 'iiif':
        
        start_download = time.time()
        src = os.getcwd()+"/data/test_imgs/sample_input/" # set the destination folder for storing mapkurator-system input
        iiif_handler = IIIFHandler(args.url, src, img_filename=img_id + '_stitched.jpg')
        iiif_handler.process_url()

        end_download = time.time()
        
        if_print_command = args.print_command
        run_img_command = "python /home/mapkurator-system/run_pipeline.py "\
        "--output_folder=/data/test_imgs/sample_output/ "\
        "--text_spotting_model_dir="+spotter_dir+" "\
        "--sample_map_path=/data/test_imgs/sample_input/ "\
        "--spotter_model="+spotter_model+" "\
        "--spotter_config="+spotter_config+" "\
        "--gpu_id="+str(gpu_id)

        exe_ret = execute_command(run_img_command, if_print_command)
        if 'error' in exe_ret:
            error = exe_ret['error']
        elif 'time_usage' in exe_ret:
            time_usage = exe_ret['time_usage']

        # read from output of stitch module and write to output folder
        input_dir =os.getcwd()+"/data/test_imgs/sample_output/intermediate_results/stitch/intermediate_results/"
        annotation_file = write_annotation(input_dir, args.dst,args.filename,"data/test_imgs/sample_input/")
        end_detection = time.time()

        print('download time: ', end_download - start_download)
        print('detection time: ', end_detection - end_download)


    if args.subcommand == 'file':
        
        map_path = args.src
        if_print_command = args.print_command
        run_img_command = "python /home/mapkurator-system/run_pipeline.py "\
        "--output_folder=/data/test_imgs/sample_output/ "\
        "--text_spotting_model_dir="+spotter_dir+" "\
        "--sample_map_path=/data/test_imgs/sample_input/ "\
        "--spotter_model="+spotter_model+" "\
        "--spotter_config="+spotter_config+" "\
        "--gpu_id="+str(gpu_id)

        exe_ret = execute_command(run_img_command, if_print_command)
        if 'error' in exe_ret:
            error = exe_ret['error']
        elif 'time_usage' in exe_ret:
            time_usage = exe_ret['time_usage']

        # read from output of stitch module and write to output folder 
        input_dir =os.getcwd()+"/data/test_imgs/sample_output/intermediate_results/stitch/intermediate_results/"
        annotation_file = write_annotation(input_dir, args.dst,args.filename,"data/test_imgs/sample_input/")