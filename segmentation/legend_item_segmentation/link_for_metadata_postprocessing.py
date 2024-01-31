
import numpy as np
from matplotlib import pyplot as plt
import cv2
import random
import os
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
import math
import json
from datetime import datetime
from scipy import sparse
import pyvips
import shutil
import csv

#import postprocessing_for_bitmap_worker as postprocessing_for_bitmap_worker


#import multiprocessing
#print(multiprocessing.cpu_count())
#PROCESSES = 8


solution_dir='LINK_Intermediate/' # set path to metadata-preprocessing output

path_to_tif = 'input.tif'
path_to_json = 'input.json'
target_map_name = 'intput'

target_dir_img = 'LOAM_Intermediate/data/cma/imgs'
target_dir_mask = 'LOAM_Intermediate/data/cma/masks'

target_dir_img_small = 'LOAM_Intermediate/data/cma_small/imgs'
target_dir_mask_small = 'LOAM_Intermediate/data/cma_small/masks'

performance_evaluation = True

crop_size = 1024
step_size = 32

shift_leftward = 300
shift_upward = 100

def dir_setting():
    if not os.path.exists(os.path.join('LOAM_Intermediate')):
        os.makedirs(os.path.join('LOAM_Intermediate'))
    if not os.path.exists(os.path.join('LOAM_Intermediate', 'data')):
        os.makedirs(os.path.join('LOAM_Intermediate', 'data'))
    if not os.path.exists(os.path.join('LOAM_Intermediate', 'data', 'cma')):
        os.makedirs(os.path.join('LOAM_Intermediate', 'data', 'cma'))
    if not os.path.exists(os.path.join('LOAM_Intermediate', 'data', 'cma_small')):
        os.makedirs(os.path.join('LOAM_Intermediate', 'data', 'cma_small'))

    if not os.path.exists(target_dir_img):
        os.makedirs(target_dir_img)
        os.makedirs(os.path.join(target_dir_img, 'sup'))
    if not os.path.exists(target_dir_mask):
        os.makedirs(target_dir_mask)
        os.makedirs(os.path.join(target_dir_mask, 'sup'))
    if not os.path.exists(target_dir_img_small):
        os.makedirs(target_dir_img_small)
        os.makedirs(os.path.join(target_dir_img_small, 'sup'))
    if not os.path.exists(target_dir_mask_small):
        os.makedirs(target_dir_mask_small)
        os.makedirs(os.path.join(target_dir_mask_small, 'sup'))

    print('Step ( 5/10): Serializing auxiliary information to support legend-item recognition...')
    print('Set up directories in "LOAM_Intermediate/data"')





def generate_auxiliary_info(target_img, target_map_name, target_category):
    img = np.zeros((crop_size*math.ceil(target_img.shape[0]/crop_size), crop_size*math.ceil(target_img.shape[1]/crop_size)), dtype='uint8')
    img[0:target_img.shape[0], 0:target_img.shape[1]] = target_img[:, :]
    img[img > 0] = 1
    

    highlighted_ratio_r = []
    for r in range(0, math.ceil(img.shape[0]/step_size)):
        r_0 = r*step_size
        r_1 = min(r*step_size+step_size, img.shape[0])
        this_highlighted_ratio = np.mean(img[r_0:r_1, :])*10.0
        highlighted_ratio_r.append(this_highlighted_ratio)
    
    highlighted_ratio_c = []
    for c in range(0, math.ceil(img.shape[1]/step_size)):
        c_0 = c*step_size
        c_1 = min(c*step_size+step_size, img.shape[1])
        this_highlighted_ratio = np.mean(img[:, c_0:c_1])*10.0
        highlighted_ratio_c.append(this_highlighted_ratio)

    #print(highlighted_ratio_r)
    #print(highlighted_ratio_c)
    if os.path.isfile('LOAM_Intermediate/data/auxiliary_info_source.csv') == False:
        with open('LOAM_Intermediate/data/auxiliary_info_source.csv','w') as fd:
            fd.write('Map_Name,Category,R/C,Values\n')
            fd.close()

    with open('LOAM_Intermediate/data/auxiliary_info_source.csv','a') as fd:
        fd.write(str(target_map_name)+','+str(target_category))
        fd.write(',R')
        for this_ratio in highlighted_ratio_r:
            fd.write(','+str(this_ratio))
        fd.write(',C')
        for this_ratio in highlighted_ratio_c:
            fd.write(','+str(this_ratio))
        fd.write('\n')
        fd.close()

    return highlighted_ratio_r, highlighted_ratio_c
    



def postprocessing_for_bitmap_worker_multiple_image(this_map_name, data_dir, groundtruth_dir, target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small, crop_size=1024, performance_evaluation=True):
    source_path_0 = os.path.join(data_dir, this_map_name, 'area_crop_binary.tif') # region of interest
    target_path_0 = os.path.join(target_dir_img, this_map_name+'.png')
    source_path_1a = os.path.join(groundtruth_dir, this_map_name+'_poly_legend_item.tif') # groundtruth
    source_path_1b = os.path.join(groundtruth_dir, this_map_name+'_ptln_legend_item.tif') # groundtruth
    target_path_1a = os.path.join(target_dir_mask, this_map_name+'_mask.png')
    target_path_1b = os.path.join(target_dir_mask, this_map_name+'_mask2.png')

    
    source_path_2 = os.path.join(data_dir, this_map_name, 'intermediate4', this_map_name+'_mask_map_key.tif') # input segmentation (input polygon candidate)
    target_path_2 = os.path.join(target_dir_img, 'sup', str(this_map_name+"_sup_0.png"))
    source_path_3 = os.path.join(data_dir, this_map_name, 'intermediate4', this_map_name+'_mask_colored.tif') # color thresholding
    target_path_3 = os.path.join(target_dir_img, 'sup', str(this_map_name+"_sup_1.png"))
    source_path_4 = os.path.join(data_dir, this_map_name, 'intermediate4', this_map_name+'_mask_black.tif') # color thresholding
    target_path_4 = os.path.join(target_dir_img, 'sup', str(this_map_name+"_sup_2.png"))
    source_path_5 = os.path.join(data_dir, this_map_name, 'intermediate3', this_map_name+'_tesseract_mask_buffer_v1.tif') # text spotting
    target_path_5 = os.path.join(target_dir_img, 'sup', str(this_map_name+"_sup_3.png"))
    source_path_6 = os.path.join(data_dir, this_map_name, 'intermediate3_2', this_map_name+'_mapkurator_mask_buffer_v1.tif') # text spotting
    target_path_6 = os.path.join(target_dir_img, 'sup', str(this_map_name+"_sup_4.png"))

    #source_path_ext = os.path.join(data_dir2, this_map_name+'_expected_crop_region.png')
    #shutil.copyfile(source_path_0, target_path_0)
    #if performance_evaluation == True:
        #shutil.copyfile(source_path_1, target_path_1)
    #shutil.copyfile(source_path_8, target_path_8)

    

    figure_info = []
    figure_info.append([source_path_1a]) # 0
    figure_info.append([source_path_1b]) # 1
    figure_info.append([source_path_0]) # 2
    figure_info.append([source_path_2]) # 3
    figure_info.append([source_path_3]) # 4
    figure_info.append([source_path_4]) # 5
    figure_info.append([source_path_5]) # 6
    figure_info.append([source_path_6]) # 7


    ratio_r = []
    ratio_c = []
    img0 = cv2.imread(figure_info[3][0])
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    this_ratio_r, this_ratio_c = generate_auxiliary_info(img0, this_map_name, 'Thresholding_1')
    ratio_r.append(this_ratio_r)
    ratio_c.append(this_ratio_c)

    img0 = cv2.imread(figure_info[4][0])
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    this_ratio_r, this_ratio_c = generate_auxiliary_info(img0, this_map_name, 'Thresholding_2')
    ratio_r.append(this_ratio_r)
    ratio_c.append(this_ratio_c)

    img01 = cv2.imread(figure_info[6][0])
    img01 = cv2.cvtColor(img01, cv2.COLOR_BGR2GRAY)
    img02 = cv2.imread(figure_info[7][0])
    img02 = cv2.cvtColor(img02, cv2.COLOR_BGR2GRAY)
    img03 = cv2.bitwise_or(img01, img02)
    img04 = cv2.bitwise_and(img02, (255-img01))
    this_ratio_r, this_ratio_c = generate_auxiliary_info(img03, this_map_name, 'Text_1')
    ratio_r.append(this_ratio_r)
    ratio_c.append(this_ratio_c)
    this_ratio_r, this_ratio_c = generate_auxiliary_info(img04, this_map_name, 'Text_2')
    ratio_r.append(this_ratio_r)
    ratio_c.append(this_ratio_c)

    ratio_r = np.array(ratio_r)
    ratio_c = np.array(ratio_c)


    img = cv2.imread(figure_info[2][0])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if os.path.isfile('LOAM_Intermediate/data/auxiliary_info_1.csv') == False:
        with open('LOAM_Intermediate/data/auxiliary_info_1.csv','w') as fd:
            fd.write('Map_Name,R,C')
            for j in range(0, 256): # 256 # 128
                fd.write(',Value_'+str(j))
            fd.write('\n')
            fd.close()
    if os.path.isfile('LOAM_Intermediate/data/auxiliary_info_2.csv') == False:
        with open('LOAM_Intermediate/data/auxiliary_info_2.csv','w') as fd:
            fd.write('Map_Name,R,C')
            for j in range(0, 4096): # 4096 # 1024
                fd.write(',Value_'+str(j))
            fd.write('\n')
            fd.close()
    for r in range(0,math.ceil(img.shape[0]/crop_size)):
        for c in range(0,math.ceil(img.shape[1]/crop_size)):
            r_0 = int(r*(crop_size/step_size))
            r_1 = int(r*(crop_size/step_size)+(crop_size/step_size))
            c_0 = int(c*(crop_size/step_size))
            c_1 = int(c*(crop_size/step_size)+(crop_size/step_size))

            with open('LOAM_Intermediate/data/auxiliary_info_1.csv','a') as fd:
                fd.write(str(this_map_name)+','+str(r)+','+str(c))
                for g in range(0, 4):
                    for this_ratio in ratio_r[g][r_0:r_1]:
                        fd.write(','+str(this_ratio))
                    for this_ratio in ratio_c[g][c_0:c_1]:
                        fd.write(','+str(this_ratio))
                fd.write('\n')
                fd.close()
            with open('LOAM_Intermediate/data/auxiliary_info_2.csv','a') as fd:
                fd.write(str(this_map_name)+','+str(r)+','+str(c))
                for g in range(0, 4):
                    for this_ratio_rr in ratio_r[g][r_0:r_1]:
                        for this_ratio_cc in ratio_c[g][c_0:c_1]:
                            this_ratio = this_ratio_rr*this_ratio_cc
                            fd.write(','+str(this_ratio))
                fd.write('\n')
                fd.close()



    print('Step ( 6/10): Cropping raster map to support legend-item recognition...')
    for this_img in range(0, len(figure_info)+4):
        if this_img == 0 or this_img == 1:
            if performance_evaluation == False:
                continue
        
        if this_img < len(figure_info):
            img = cv2.imread(figure_info[this_img][0])
        elif this_img < len(figure_info)+2:
            img = cv2.imread(figure_info[6][0])
        else:
            img = cv2.imread(figure_info[7][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if this_img >= len(figure_info):
            img_extended = np.zeros((img.shape[0]+shift_upward, img.shape[1]+shift_leftward), dtype='uint8')
            img_extended[0:img.shape[0], 0:img.shape[1]] = img[:, :]
        
        empty_grid = np.zeros((crop_size, crop_size), dtype='uint8').astype(float)

        for r in range(0,math.ceil(img.shape[0]/crop_size)):
            for c in range(0,math.ceil(img.shape[1]/crop_size)):
                if this_img == 0:
                    this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+str(r)+'_'+str(c)+'_mask.png'))
                elif this_img == 1:
                    this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+str(r)+'_'+str(c)+'_mask2.png'))
                elif this_img == 2:
                    this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+str(r)+'_'+str(c)+'.png'))
                else:
                    this_output_file_name = os.path.join(target_dir_img_small, 'sup', str(this_map_name+'_'+str(r)+'_'+str(c)+'_sup_'+str(this_img-3)+'.png'))
                
                if (min(r*crop_size+crop_size, img.shape[0]) - r*crop_size <= 0) or (min(c*crop_size+crop_size, img.shape[1]) - c*crop_size <= 0):
                    continue
                
                r_0 = r*crop_size
                r_1 = min(r*crop_size+crop_size, img.shape[0])
                c_0 = c*crop_size
                c_1 = min(c*crop_size+crop_size, img.shape[1])

                #print(r, c, r_0, r_1, c_0, c_1)
                if this_img < len(figure_info):
                    if r_1-r_0 < crop_size or c_1-c_0 < crop_size:
                        if r_1-r_0 < crop_size:
                            img_concat_temp = np.concatenate([img[r_0:r_1, c_0:c_1], empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0)]], axis=0)
                            #print(img[r_0:r_1, c_0:c_1].shape, empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0)].shape, img_concat_temp.shape)
                        else:
                            img_concat_temp = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
                            #print(img[r_0:r_1, c_0:c_1].shape, img_concat_temp.shape)
                        if c_1-c_0 < crop_size:
                            img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0)], empty_grid[:, 0:crop_size-(c_1-c_0)]], axis=1)
                            #print(img_concat_temp[:, :(c_1-c_0)].shape, empty_grid[:, 0:crop_size-(c_1-c_0)].shape, img_concat.shape)
                        else:
                            img_concat = np.copy(img_concat_temp).astype(float)
                            #print(img_concat_temp.shape, img_concat.shape)
                    else:
                        img_concat = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
                else:
                    if this_img%2 == 0:
                        r_0 += shift_upward
                        r_1 += shift_upward
                    elif this_img%2 == 1:
                        c_0 += shift_leftward
                        c_1 += shift_leftward
                    
                    if r_1-r_0 < crop_size or c_1-c_0 < crop_size:
                        if r_1-r_0 < crop_size:
                            img_concat_temp = np.concatenate([img_extended[r_0:r_1, c_0:c_1], empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0)]], axis=0)
                            #print(img[r_0:r_1, c_0:c_1].shape, empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0)].shape, img_concat_temp.shape)
                        else:
                            img_concat_temp = np.copy(img_extended[r_0:r_1, c_0:c_1]).astype(float)
                            #print(img[r_0:r_1, c_0:c_1].shape, img_concat_temp.shape)
                        if c_1-c_0 < crop_size:
                            img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0)], empty_grid[:, 0:crop_size-(c_1-c_0)]], axis=1)
                            #print(img_concat_temp[:, :(c_1-c_0)].shape, empty_grid[:, 0:crop_size-(c_1-c_0)].shape, img_concat.shape)
                        else:
                            img_concat = np.copy(img_concat_temp).astype(float)
                            #print(img_concat_temp.shape, img_concat.shape)
                    else:
                        img_concat = np.copy(img_extended[r_0:r_1, c_0:c_1]).astype(float)

                

                if this_img == 0 or this_img == 1:
                    img_concat[img_concat > 0] = 1
                    #img_concat[img_concat > 0] = 255
                else:
                    img_concat[img_concat > 0] = 255
                
                cv2.imwrite(this_output_file_name, img_concat)

                if this_img == 0 or this_img == 1:
                    img_concat[img_concat > 0] = 255
                    if this_img == 0:
                        this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+str(r)+'_'+str(c)+'_mask_vis.png'))
                    if this_img == 1:
                        this_output_file_name = os.path.join(target_dir_mask_small, str(this_map_name+'_'+str(r)+'_'+str(c)+'_mask2_vis.png'))
                    cv2.imwrite(this_output_file_name, img_concat)
                    
    
    return (math.ceil(img.shape[0]/crop_size) * math.ceil(img.shape[1]/crop_size))



def start_data_postprocessing(input_this_map_name, input_data_dir, input_groundtruth_dir, input_target_dir_img, input_target_dir_mask, input_target_dir_img_small, input_target_dir_mask_small, input_crop_size, input_performance_evaluation):
    dir_setting()
    postprocessing_for_bitmap_worker_multiple_image(input_this_map_name, input_data_dir, input_groundtruth_dir, input_target_dir_img, input_target_dir_mask, input_target_dir_img_small, input_target_dir_mask_small, input_crop_size, input_performance_evaluation)


def main():
    dir_setting()


    missing_list = []
    with open('missing.csv', newline='') as fdd:
        reader = csv.reader(fdd)
        for row in reader:
            missing_list.append(row[0])
    print(missing_list)


    runningtime_list = []
    global_runningtime_start = datetime.now()
    runningtime_start = datetime.now()
    this_map_count = 0
    total_map_count = len(os.listdir('Data/validation/'))/2
    for target_map_cand in os.listdir('Data/validation/'):
        if '.tif' in target_map_cand:
            this_runningtime_start = datetime.now()
            target_map = target_map_cand.split('.tif')[0]
            if target_map_cand in missing_list:
                this_map_count += 1
                print('Disintegrity map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                runningtime_list.append([target_map, datetime.now()-this_runningtime_start])
                continue
            
            postprocessing_for_bitmap_worker_multiple_image(target_map, 'LINK_Intermediate/validation', 'Data/validation_groundtruth', target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small, crop_size=1024, performance_evaluation=True)
            this_map_count += 1
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count)+'   for validation dataset...'+str(datetime.now()-runningtime_start)+'   for full dataset...'+str(datetime.now()-global_runningtime_start))
            runningtime_list.append([target_map, datetime.now()-this_runningtime_start])

    
    runningtime_start = datetime.now()
    this_map_count = 0
    total_map_count = len(os.listdir('Data/testing/'))/2
    for target_map_cand in os.listdir('Data/testing/'):
        if '.tif' in target_map_cand:
            this_runningtime_start = datetime.now()
            target_map = target_map_cand.split('.tif')[0]
            if target_map_cand in missing_list:
                this_map_count += 1
                print('Disintegrity map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                runningtime_list.append([target_map, datetime.now()-this_runningtime_start])
                continue

            postprocessing_for_bitmap_worker_multiple_image(target_map, 'LINK_Intermediate/testing', 'Data/testing_groundtruth', target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small, crop_size=1024, performance_evaluation=True)
            this_map_count += 1
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count)+'   for testing dataset...'+str(datetime.now()-runningtime_start)+'   for full dataset...'+str(datetime.now()-global_runningtime_start))
            runningtime_list.append([target_map, datetime.now()-this_runningtime_start])

    
    runningtime_start = datetime.now()
    this_map_count = 0
    total_map_count = len(os.listdir('Data/training/'))/2
    for target_map_cand in os.listdir('Data/training/'):
        if '.tif' in target_map_cand:
            this_runningtime_start = datetime.now()
            target_map = target_map_cand.split('.tif')[0]
            if target_map_cand in missing_list:
                this_map_count += 1
                print('Disintegrity map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                runningtime_list.append([target_map, datetime.now()-this_runningtime_start])
                continue

            postprocessing_for_bitmap_worker_multiple_image(target_map, 'LINK_Intermediate/training', 'Data/training_groundtruth', target_dir_img, target_dir_mask, target_dir_img_small, target_dir_mask_small, crop_size=1024, performance_evaluation=True)
            this_map_count += 1
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count)+'   for training dataset...'+str(datetime.now()-runningtime_start)+'   for full dataset...'+str(datetime.now()-global_runningtime_start))
            runningtime_list.append([target_map, datetime.now()-this_runningtime_start])

    
    
    if os.path.isfile('LOAM_Intermediate/data/runningtime_record.csv') == False:
        with open('LOAM_Intermediate/data/runningtime_record.csv','w') as fd:
            fd.write('Map_Name,Running_Time\n')
            fd.close()

    with open('LOAM_Intermediate/data/runningtime_record.csv','a') as fd:
        for running_time_info in runningtime_list:
            fd.write(str(running_time_info[0])+','+str(running_time_info[1])+'\n')
        fd.close()
    


    

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()



