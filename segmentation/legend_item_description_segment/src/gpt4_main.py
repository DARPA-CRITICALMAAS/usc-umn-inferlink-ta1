import json
import os
import cv2
import random
import numpy as np
from utils import read_poly_roi_from_json, read_ln_pt_roi_from_json, read_symbol_bbox_from_json, read_legend_json
from input_process import generate_pts4group, group_by_columns
from input_process import remove_fp_bbox
from gpt4_input_generation import generate_gpt4_input
from symbol_description_extraction_gpt4 import gpt_extract_symbol_description
from symbol_bbox_ocr import get_symbol_names
from symbol_bbox_ocr import match_orc_gpt_results
import logging
import threading
import time
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--map_dir',
                    type=str,
                   default='/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps')
parser.add_argument('--legend_json_dir',
                   type=str,
                   default='/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps_legend_outputs')
parser.add_argument('--symbol_json_dir',
                   type=str,
                   default='/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps_legend_item_outputs')
parser.add_argument('--map_name',
                   type=str,
                   default='54565_18569')
parser.add_argument('--temp_dir',
                  type=str,
                  default='/data/weiweidu/layout_segment_v2/temp')
parser.add_argument('--output_dir',
                   type=str,
                   default='/data/weiweidu/temp')
parser.add_argument('--log_path',
                  type=str,
                  default='./item_description_logger.log')

args = parser.parse_args()
logger = logging.getLogger('item_description_logger')
handler = logging.FileHandler(f'{args.log_path}', mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

def heartbeat(heartbeat_interval, stop_event):
    try:
        while not stop_event.is_set():  # Continue until the stop event is set
            logger.info("Heartbeat: GPT4 API is still running...")
            time.sleep(heartbeat_interval)
    except KeyboardInterrupt:
        logger.info('Application stopped.')

def long_running_task(duration=5):
    # Simulate a long-running task
    for _ in range(duration):
#         logger.info("GPT4 API is still working...")
        time.sleep(1)

def run_item_description_extraction(legend_dir, legend_item_dir, map_name, map_dir, \
                                    intermediate_dir, output_dir, legend_type='polygon', max_gpt_attempt=3):
   
    legend_item_path = os.path.join(symbol_json_dir, \
                                    map_name+('_PolygonType.geojson' if legend_type=='polygon' \
                                              else '_PointLineType.geojson'))
    
    if not os.path.exists(legend_item_path):
        all_symbol_bboxes = []
    else:
        all_symbol_bboxes = read_symbol_bbox_from_json(legend_item_dir, map_name)
    
    roi_list = []
    
    if legend_type == 'polygon':
        roi_list = read_poly_roi_from_json(legend_dir, map_name+'_map_segmentation.json')
    if legend_type == 'line': 
        roi_list = read_ln_pt_roi_from_json(legend_dir, map_name+'_map_segmentation.json')
    if legend_type == 'point': 
        roi_list = read_ln_pt_roi_from_json(legend_dir, map_name+'_map_segmentation.json')
    

    map_path = os.path.join(map_dir, map_name+'.tif')
    map_tif = cv2.imread(map_path)
    
    logger.info(f'Processing {map_name}.tif')
    
    all_matched_symbol_descr, all_gpt_res, all_bbox_ocr = {}, {}, {}
    
    for i, legend_area in enumerate(roi_list):
        x, y, w, h = legend_area
        logger.info(f'Processing the {i+1}/{len(roi_list)} legend area {legend_area}')
        pts, bboxes = generate_pts4group(legend_area, all_symbol_bboxes) 
        
        if len(pts) == 0:
            logger.warning(f'No extracted legend item bounding boxes in the legend_area')
            refined_grouped_bbox = {}
#             continue
        else:
            grouped_pts, grouped_bboxes = group_by_columns(pts, bboxes, tolerance=40.0)

            logger.info(f'The legend area has {len(grouped_pts.keys())} columns')

            refined_grouped_bbox = remove_fp_bbox(grouped_bboxes, map_tif)

            logger.info(f'After refining, the legend area has {len(refined_grouped_bbox.keys())} columns')

            if len(refined_grouped_bbox.keys()) == 0:
                logger.warning(f'After refining, no extracted legend item bounding boxes in the legend_area')
#                 continue
        
        col_gpt4_input_path = generate_gpt4_input(refined_grouped_bbox, legend_area, map_tif, map_name, intermediate_dir)       
        
        for i, col_cent in enumerate(col_gpt4_input_path.keys()):
            for img_name in col_gpt4_input_path[col_cent]:
                img_path = os.path.join(intermediate_dir, img_name)
                if not os.path.exists(img_path):
                    logger.warning(f'Image file not exists {img_path}')
                    continue
                logger.info(f'GPT input image name {img_name}')
                width = int(img_name.split('.')[0].split('_')[-2])
                logger.info(f'The GPT4 is processing {i+1}/{len(col_gpt4_input_path.keys())} image {img_name}')

                heartbeat_interval = 20  # seconds
                task_duration = 10  # seconds, for example
                # Create an event to signal when to stop the heartbeat
                stop_event = threading.Event()
                # Start the heartbeat thread
                heartbeat_thread = threading.Thread(target=heartbeat, args=(heartbeat_interval, stop_event))
                heartbeat_thread.start()
                # Run the long-running task
                long_running_task(task_duration)

                for attempt in range(max_gpt_attempt):
                    logger.info(f'The GPT4 Attempt {attempt+1}/{max_gpt_attempt}')
                    try:
                        gpt_json_res = gpt_extract_symbol_description(img_path, attempt)
                        logger.info(f'The GPT4 done {i+1}/{len(col_gpt4_input_path.keys())} images in the legend area')
                        break
                    except Exception as error:
                        logger.warning(f'The GPT4 error: {error}')
                        gpt_json_res = None
    #             gpt_json_res = None

               # Once the task is done, signal the heartbeat thread to stop and wait for it to finish
                stop_event.set()
                heartbeat_thread.join()

                if not gpt_json_res:
                    logger.warning(f'The GPT4 results are empty')


                logger.info(f'OCR processing in image')

                bbox_ocr_dict = get_symbol_names(map_tif, refined_grouped_bbox[col_cent], width) if refined_grouped_bbox!={} else {}

                logger.info(f'Matching the results from OCR and GPT4')

                matched_item_descr = match_orc_gpt_results(bbox_ocr_dict, gpt_json_res, map_tif)

                if matched_item_descr:
                    all_matched_symbol_descr = {**all_matched_symbol_descr, **matched_item_descr}
                if gpt_json_res:
                    all_gpt_res = {**all_gpt_res, **gpt_json_res}
                if bbox_ocr_dict:
                    all_bbox_ocr = {**all_bbox_ocr, **bbox_ocr_dict}
    
    map_content_bbox, poly_bbox, ptln_bbox = read_legend_json(legend_dir, map_name)

    all_matched_symbol_descr['map_content_box'] = map_content_bbox
    all_matched_symbol_descr['poly_box'] = poly_bbox
    all_matched_symbol_descr['ptln_box'] = ptln_bbox
    all_matched_symbol_descr['map_dimension'] = map_tif.shape[:2]
    
    if not all_gpt_res:
        all_matched_symbol_descr['provenance'] = 'ocr'
    else:
        all_matched_symbol_descr['provenance'] = 'ocr_gpt'
    
    if legend_type in ['line', 'point']:
        gpt_output4points = {}
        gpt_output4points['map_content_box'] = map_content_bbox
        gpt_output4points['poly_box'] = poly_bbox
        gpt_output4points['ptln_box'] = ptln_bbox
        gpt_output4points['map_dimension'] = map_tif.shape[:2]
        
        for k, v in all_gpt_res.items():
            gpt_output4points[str(random.sample(range(1, 10000), 4))] = {"description": v, "symbol name": k}
        
        with open(f'{output_dir}/{map_name}_gpt_{legend_type}.json', "w") as outfile:
            json.dump(gpt_output4points, outfile) 

    with open(f"{output_dir}/{map_name}_{legend_type}.json", "w") as outfile:
        json.dump(all_matched_symbol_descr, outfile)
        
    logger.info(f'Save the results in {output_dir}/{map_name}_{legend_type}.json')
    return

if __name__ == '__main__':
    args = parser.parse_args()
    
    map_dir = args.map_dir
    legend_json_dir = args.legend_json_dir
    symbol_json_dir = args.symbol_json_dir
    temp_dir = args.temp_dir
    output_dir = args.output_dir
    map_name = args.map_name
    
    if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for legend_type in ['polygon', 'line',  'point']:
        # check the existence of input files
        legend_path = os.path.join(legend_json_dir, map_name+'_map_segmentation.json')
        legend_item_path = os.path.join(symbol_json_dir, \
                                        map_name+('_PolygonType.geojson' if legend_type=='polygon' \
                                                  else '_PointLineType.geojson'))
        map_path = os.path.join(map_dir, map_name+'.tif')

        if not os.path.exists(legend_path):
            logger.error(f'No results from legend segment module.')
            continue
        
        if not os.path.exists(legend_item_path):
            logger.warning(f'No results from legend item module.')
        
        if not os.path.exists(map_path):
            logger.error(f'The map does not exist at {map_path}.')
            continue

        run_item_description_extraction(legend_json_dir, symbol_json_dir,map_name, map_dir, \
                                        temp_dir, output_dir, legend_type=legend_type)
