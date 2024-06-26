import time
import os
import sys
import cv2
import json
import numpy as np
import logging
from argparse import ArgumentParser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = ArgumentParser()
parser.add_argument('--input_dir', default='/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps', help='a folder has testing maps')
parser.add_argument('--legend_dir', default='/data/weiweidu/criticalmaas_data/hackathon2/more_nickle_maps_legend_outputs_', help='a output folder from the legend segment module')
parser.add_argument('--map_name', default='10705_61989', help='testing map name')
parser.add_argument('--patch_sizes', type=int, nargs='+', help='a list of patch size')
parser.add_argument('--strides', type=int, nargs='+', \
                    help='a list of stride, the length is the same as path_sizes')
parser.add_argument('--only_crop_map_area', type=str2bool, nargs='+', \
                    help='a list of T/F, the length is the same as path_sizes')
parser.add_argument('--output_dir', default='/data/weiweidu', help='a folder to save cropped images')
parser.add_argument('--log_path', type=str, default='./map_crop_logger.log')

args = parser.parse_args()

logger = logging.getLogger('map_crop_logger')
handler = logging.FileHandler(f'{args.log_path}', mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')

def crop_map(map_image, map_name, map_area_bbox, patch_size, stride, output_dir, only_crop_map_area=True):
    
    h, w = map_image.shape[:2]
    
    if map_area_bbox is not None and only_crop_map_area:
        map_content_y, map_content_x, map_content_w, map_content_h = map_area_bbox
        map_content_mask = np.zeros((h, w))
        cv2.rectangle(map_content_mask, (int(map_content_y),int(map_content_x)), \
                      (int(map_content_y)+int(map_content_w),int(map_content_x)+int(map_content_h)), 1, -1)
    elif map_area_bbox is not None and not only_crop_map_area: 
        #no extracted legend area, crop images outside the map content area
        map_content_y, map_content_x, map_content_w, map_content_h = map_area_bbox
        map_content_mask = np.zeros((h, w))
        cv2.rectangle(map_content_mask, (int(map_content_y),int(map_content_x)), \
                      (int(map_content_y)+int(map_content_w),int(map_content_x)+int(map_content_h)), 1, -1)
        map_content_mask = 1 - map_content_mask # the mask for area outside the map content area
    else:
        map_content_mask = np.ones((h, w))
    
    p_h, p_w = patch_size, patch_size
    # pad_h, pad_w = 5, 5
    num_h = h // stride
    num_w = w // stride
    
    if map_area_bbox is not None and only_crop_map_area:
        output_folder = os.path.join(output_dir, f'{map_name}_g{patch_size}_s{stride}_wo_legend')
    else:
        output_folder = os.path.join(output_dir, f'{map_name}_g{patch_size}_s{stride}_wo_legend')
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    x_ = np.int32(np.linspace(5, h-5-p_h, num_h)) 
    y_ = np.int32(np.linspace(5, w-5-p_w, num_w)) 
    
    ind = np.meshgrid(x_, y_, indexing='ij')

    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
        patch_mask = map_content_mask[start[0]:start[0]+p_h, start[1]:start[1]+p_w]
        if np.sum(patch_mask) < 200:
            continue
        patch = map_image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :]

        output_path = os.path.join(output_folder, f'{map_name}_{start[0]}_{start[1]}.png')
            
        cv2.imwrite(output_path, patch)
    
    return output_folder

def read_map_content_area_from_json(legend_json_path):# Use glob.glob() to find files matching the pattern
    map_area_bbox = None
    ptln_legend_area_bbox = None
    try:   
        with open(legend_json_path, 'r', encoding='utf-8') as file:
            legend_dict = json.load(file)
    except:
        return FileNotFoundError(f'{json_path} does not exist')

    for item in legend_dict['segments']:
        if 'map' == item['class_label']:
            map_area_bbox = item['bbox'] 
        if 'legend_points_lines' == item['class_label']:
            ptln_legend_area_bbox = item['bbox']
    return map_area_bbox, ptln_legend_area_bbox

def crop_map_main(args):    
    input_path = os.path.join(args.input_dir, args.map_name+'.tif')
    map_image = cv2.imread(input_path)
    
    legend_json_path = os.path.join(args.legend_dir, args.map_name+'_map_segmentation.json')

    try:
        map_area_bbox, ptln_legend_area_bbox = read_map_content_area_from_json(legend_json_path)
        if map_area_bbox:
            logger.info(f'Get the map content bounding box: {map_area_bbox}')
        else:
            logger.warning(f'Legend segmentation json does not extract the map content area')
    except:
        map_area_bbox, ptln_legend_area_bbox = None, None
        logger.warning(f'Legend segmentation json does not exist in {legend_json_path}')
        
    for i, patch_size in enumerate(args.patch_sizes[:-1]):
        logger.info(f'generating patch_size={patch_size} for {args.map_name}')
        s_time = time.time()
        
        map_crop_output_path = crop_map(map_image, args.map_name, map_area_bbox, \
                               patch_size, args.strides[i], args.output_dir, args.only_crop_map_area[i])
        e_time = time.time()
        logger.info(f'Processing time {e_time-s_time}s')
        logger.info(f'Saved the cropped images for {args.map_name} in {map_crop_output_path}')

    # crop the legend area
    patch_size, stride_size = args.patch_sizes[-1], args.strides[-1]
    logger.info(f'generating patch_size={patch_size} for the point and line legend areas for {args.map_name}')
    
    if ptln_legend_area_bbox is not None:
        legend_crop_output_path = crop_map(map_image, args.map_name, ptln_legend_area_bbox, \
                                   patch_size, stride_size, args.output_dir, True)
    else:
        legend_crop_output_path = crop_map(map_image, args.map_name, map_area_bbox, \
                                   patch_size, stride_size, args.output_dir, False)
    logger.info(f'Saved the cropped images for {args.map_name} in {legend_crop_output_path}')
    return

if __name__ == '__main__':
    args = parser.parse_args()
    
    #sanity check
    if not (len(args.patch_sizes) == len(args.strides) == len(args.only_crop_map_area)):
        raise ValueError("patch_sizes, strides, only_crop_map_area arguments must have the same length")
    
    if not os.path.exists(os.path.join(args.input_dir, args.map_name+'.tif')):
        raise FileNotFoundError(f'The tif map does not exist')
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
  
    crop_map_main(args)
    sys.exit(0)
        
