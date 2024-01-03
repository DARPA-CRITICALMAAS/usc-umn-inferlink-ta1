import time
import os
import cv2
import numpy as np

def crop_map(map_image, map_name, patch_size, stride, output_dir):
    h, w = map_image.shape[:2]
    p_h, p_w = patch_size, patch_size
    pad_h, pad_w = 5, 5
    num_h = h // stride
    num_w = w // stride
    
    output_folder = os.path.join(output_dir, f'{map_name}_g{patch_size}_s{stride}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    x_ = np.int32(np.linspace(5, h-5-p_h, num_h)) 
    y_ = np.int32(np.linspace(5, w-5-p_w, num_w)) 
    
    ind = np.meshgrid(x_, y_, indexing='ij')

    for i, start in enumerate(list(np.array(ind).reshape(2,-1).T)):
        
        patch = np.pad(map_image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, :],\
                       ((pad_h,pad_h),(pad_w,pad_w),(0,0)), mode='constant')
        
        output_path = os.path.join(output_folder, f'{map_name}_{start[0]}_{start[1]}.png')
        cv2.imwrite(output_path, patch)
    
    return output_folder

def crop_map_main(args):    
    input_path = os.path.join(args.input_dir, args.map_name+'.tif')
    map_image = cv2.imread(input_path)
    
    for i, patch_size in enumerate(args.patch_sizes):
        print(f'*** generating {patch_size} for {args.map_name} ***')
        s_time = time.time()
        output_path = crop_map(map_image, args.map_name, patch_size, args.strides[i], args.output_dir)
        
        e_time = time.time()
        print(f'processing time {e_time-s_time}s')
        print(f'*** saved the cropped images for {args.map_name} in {output_path}')
        
if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input_dir', default=None, help='a folder has testing maps')
    parser.add_argument('--map_name', default=None, help='testing map name')
    parser.add_argument('--patch_sizes', metavar='N', type=int, nargs='+', help='a list of patch size')
    parser.add_argument('--strides', metavar='N', type=int, nargs='+', \
                        help='a list of stride, the length is the same as path_sizes')
    parser.add_argument('--output_dir', default=None, help='a folder to save cropped images')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    crop_map_main(args)
        
        
