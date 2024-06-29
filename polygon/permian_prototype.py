import cv2
import numpy as np
import os
import json
import csv
import shutil
from datetime import datetime
import math


import multiprocessing
PROCESSES = 10

import worker.generating_colorization_groundtruth as generating_colorization_groundtruth
import worker.map_area_segmenter as map_area_segmenter

#from worker.linking_legend_item_description import integrate_linking_csv as integrating_auxiliary_info
import worker.segmenting_legend_item as segmenting_legend_item
import worker.constructing_region_map as constructing_region_map
import worker.merging_region_map as merging_region_map
import worker.assigning_region_map as assigning_region_map
import worker.merging_from_bitmap as merging_from_bitmap
import worker.thresholding_with_conditions as thresholding_with_conditions
import worker.reading_base_for_integration as reading_base_for_integration

import worker.postprocessing_for_bitmap as postprocessing_for_bitmap_worker
from worker.postprocessing_for_bitmap import postprocessing_for_bitmap as postprocessing_for_bitmap
import worker.getting_color_groundtruth as getting_color_groundtruth
#from worker.linking_legend_item_description import bert_processing as bert_processing

#from worker.extracting_legend_item_description import extracting_legend_item_description as extracting_lid
#from worker.linking_legend_item_description import linking_legend_item_description as linking_lid

import worker.color_replacement as color_replacement

category = ['testing', 'validation', 'training']

def directory_setting(category, dir_to_intermediate):
    os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General'), exist_ok=True)
    os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate'), exist_ok=True)
    for this_category in range(0, 1):
        if dir_to_intermediate is not None:
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_source'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_grayscale'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_binary'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_roi'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_info'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_text'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_text_intermediate'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_item'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_region'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'imgs'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'imgs', 'sup'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'masks'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'masks', 'sup'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'predict'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'predict', 'cma'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'predict', 'cma_small'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'predict', 'cma_small_colored'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_groundtruth'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/', 'small'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/', 'small_colored'), exist_ok=True)
            os.makedirs(os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/', 'colored'), exist_ok=True)
        else:
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_source'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_grayscale'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_binary'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_roi'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_info'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_text'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_text_intermediate'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_item'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_region'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'data'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'data', 'cma_small'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'data', 'cma_small', 'imgs'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'data', 'cma_small', 'imgs', 'sup'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'data', 'cma_small', 'masks'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'data', 'cma_small', 'masks', 'sup'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'predict'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'predict', 'cma'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'predict', 'cma_small'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_Intermediate', 'predict', 'cma_small_colored'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_groundtruth'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_replace/'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_replace/', 'small'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_replace/', 'small_colored'), exist_ok=True)
            os.makedirs(os.path.join('PERMIAN_General', str(category[this_category])+'_replace/', 'colored'), exist_ok=True)
    return True



def identifying_target_map(this_category, path_to_source_map, path_to_target_map_0, dir_to_intermediate, map_name, path_to_tif, path_to_json):
    info_list = []
    with open(path_to_json) as f:
        gj = json.load(f)

    poly_counter = 0
    legend_name_check = []
    labeling_error_flag = False
    for this_gj in gj['shapes']:
        names = this_gj['label']
        
        if '_poly' not in names:
            continue
        if names in legend_name_check:
            labeling_error_flag = True
            #print(target_map_cand, names)
            continue
        poly_counter = poly_counter+1

    if poly_counter > 0:
        # copy tif file
        shutil.copyfile(path_to_tif, os.path.join(path_to_target_map_0, map_name))

        # copy json file conditionally
        if labeling_error_flag == False:
            shutil.copyfile(path_to_json, os.path.join(path_to_target_map_0, map_name.replace('.tif', '.json')))
        else:
            print('There is labeling error in the original json file... Already manually removed the duplicated items...', map_name)

        # add to a list to support further processing
        info_list.append(map_name)

        #break
    
    with open(os.path.join(dir_to_intermediate, 'PERMIAN_General', 'map_list_'+str(category[this_category])+'.csv'),'w') as fd:
        for this_info in info_list:
            fd.write(str(this_info)+'\n')
        fd.close()
    
    return info_list



def cropping_map_area(target_map_list, path_to_source_map, path_to_target_map_3):
    info_list = []
    for target_map_cand in target_map_list:
        info_list.append([path_to_source_map, target_map_cand, os.path.join(path_to_target_map_3, target_map_cand), path_to_target_map_3])
    print(len(info_list))

    runningtime_start = datetime.now()
    map_processed = 0
    with multiprocessing.Pool(8) as pool:
        callback = pool.starmap_async(map_area_segmenter.cropping_worker, [(this_info[0], this_info[1], this_info[2], this_info[3], ) for this_info in info_list])
        multiprocessing_results = callback.get()

        for get_return, this_map_cand in multiprocessing_results:
            if get_return == True:
                map_processed += 1
                print(str(map_processed)+'/'+str(len(info_list)), str(datetime.now()-runningtime_start))
            else:
                print(this_map_cand)
    return True


def generating_color_groundtruth(target_map_list, path_to_source_map, path_to_source_groundtruth, path_to_target_map_1, path_to_target_map_2, path_to_target_map_3, path_to_target_map_4, path_to_target_groundtruth, is_training):
    info_list = []
    for target_map_cand in target_map_list:
        info_list.append([target_map_cand, path_to_source_map, path_to_target_map_1, path_to_target_map_2, path_to_source_groundtruth, path_to_target_groundtruth, path_to_target_map_3, path_to_target_map_4, is_training])
    print(len(info_list))

    runningtime_start = datetime.now()
    map_processed = 0
    with multiprocessing.Pool(10) as pool:
        callback = pool.starmap_async(generating_colorization_groundtruth.generating_colorization_groundtruth, [(this_info[0], this_info[1], this_info[2], this_info[3], this_info[4], this_info[5], this_info[6], this_info[7], this_info[8], ) for this_info in info_list])
        multiprocessing_results = callback.get()

        for get_return, this_map_cand in multiprocessing_results:
            if get_return == True:
                map_processed += 1
                print(str(map_processed)+'/'+str(len(info_list)), str(datetime.now()-runningtime_start))
            else:
                print(this_map_cand)
    return True


'''
def generating_legend_text(target_map_list, path_to_target_map_0, path_to_target_map_3, path_to_target_map_5, path_to_target_map_6, path_to_target_map_7):
    runningtime_start = datetime.now()
    map_processed = 0
    for target_map_cand in target_map_list:
        
        print('Processing extracting_legend_item_description...')
        extracting_lid(path_to_target_map_0, target_map_cand.replace('.tif', ''), path_to_target_map_0, path_to_target_map_3, path_to_target_map_6, path_to_target_map_7)
        
        print('Processing linking_legend_item_description...')
        linking_lid(target_map_cand.replace('.tif', ''), path_to_target_map_0, path_to_target_map_7, path_to_target_map_5)
        integrating_auxiliary_info(target_map_cand.replace('.tif', ''), path_to_target_map_5)
        
        print('Processing bert_processing...')
        print(target_map_cand.replace('.tif', ''), path_to_target_map_5)
        bert_processing(target_map_cand.replace('.tif', ''), path_to_target_map_5)

        map_processed += 1
        print(str(map_processed)+'/'+str(len(target_map_list)), str(datetime.now()-runningtime_start))
    print(map_processed)
    return True
'''



def cropping_legend_area(target_map_list, path_to_target_map_0, path_to_target_map_2, path_to_target_map_8):
    for target_map_cand in target_map_list:
        with open(os.path.join(path_to_target_map_0, target_map_cand.replace('.tif', '.json'))) as f:
            gj = json.load(f)

        poly_counter = 0
        legend_name = []
        legend_name_check = []

        img = cv2.imread(os.path.join(path_to_target_map_2, target_map_cand))
        legend_feature = []

        for this_gj in gj['shapes']:
            names = this_gj['label']
            features = this_gj['points']
            
            if '_poly' not in names:
                continue
            if names in legend_name_check:
                #print(target_map_cand, names)
                continue
            legend_name_check.append(names)
            legend_name.append(names)

            geoms = np.array(features)
            y_min = int(np.min(geoms, axis=0)[0])
            y_max = int(np.max(geoms, axis=0)[0])
            x_min = int(np.min(geoms, axis=0)[1])
            x_max = int(np.max(geoms, axis=0)[1])

            img_legend = np.zeros((x_max-x_min, y_max-y_min, 3), dtype='uint8')
            img_legend = np.copy(img[x_min:x_max, y_min:y_max, :])
            legend_feature.append(img_legend)

            poly_counter = poly_counter+1
        print(poly_counter)
        
        if poly_counter > 0:
            info_list = []
            for this_legend in range(0, poly_counter):
                info_list.append([path_to_target_map_2, target_map_cand, path_to_target_map_8, legend_name[this_legend], legend_feature[this_legend]])
            
            poly_processed = 0
            with multiprocessing.Pool(8) as pool:
                callback = pool.starmap_async(segmenting_legend_item.segmenting_legend_item, [(this_info[0], this_info[1], this_info[2], this_info[3], this_info[4], ) for this_info in info_list])
                multiprocessing_results = callback.get()

                for get_return, this_map, this_map_cand in multiprocessing_results:
                    if get_return == True:
                        poly_processed += 1
                    else:
                        print(this_map, this_map_cand)
            print(target_map_cand, len(info_list))
    return True



def importing_color_dict(color_info_source):
    color_dict_indexed_hsv = {}
    color_dict_indexed_rgb = {}
    color_dict_indexed_lab = {}
    item_dict_indexed = {}

    color_info = np.genfromtxt(color_info_source, delimiter=',', dtype=None, encoding='utf8')
    print('color_info shape:', color_info.shape)

    for info_index in range(1, color_info.shape[0]):
        color_dict_indexed_hsv.update({color_info[info_index, 1] : [color_info[info_index, 2].astype(float)*180.0, color_info[info_index, 3].astype(float)*255.0, color_info[info_index, 4].astype(float)*255.0]})
        color_dict_indexed_rgb.update({color_info[info_index, 1] : [color_info[info_index, 5].astype(float)*255.0, color_info[info_index, 6].astype(float)*255.0, color_info[info_index, 7].astype(float)*255.0]})
        color_dict_indexed_lab.update({color_info[info_index, 1] : [color_info[info_index, 8].astype(float)*255.0, color_info[info_index, 9].astype(float)*255.0, color_info[info_index, 10].astype(float)*255.0]})

        
        #item_dict_indexed.update({color_info[info_index, 0] : color_info[info_index, 1]})
        if str(color_info[info_index, 0]) in item_dict_indexed:
            item_dict_indexed[str(color_info[info_index, 0])].append(str(color_info[info_index, 1]))
        else:
            item_dict_indexed[str(color_info[info_index, 0])] = [str(color_info[info_index, 1])]
    print(len(color_dict_indexed_hsv), len(color_dict_indexed_rgb), len(color_dict_indexed_lab), len(item_dict_indexed))
    
    return color_dict_indexed_hsv, color_dict_indexed_rgb, color_dict_indexed_lab, item_dict_indexed



def generating_region_map(target_map_list, path_to_target_map_0, path_to_replaced_img_colored, path_to_target_map_3, path_to_target_map_4, path_to_target_map_5, path_to_cropped_img, path_to_cropped_img_sup, path_to_cropped_mask_sup, path_to_cropped_intermediate, path_to_cropped_predict, path_to_merged_predict, support_training=False, image_crop_size=1024):

    color_dict_indexed_hsv, color_dict_indexed_rgb, color_dict_indexed_lab, item_dict_indexed = importing_color_dict(os.path.join(path_to_target_map_5, 'color_info.csv'))
    #print(color_dict_indexed_hsv, color_dict_indexed_rgb, color_dict_indexed_lab,)
    #print(item_dict_indexed)

    #stratification_layer = 1
    stratification_layer = 3

    for target_map_cand in target_map_list:


        
        max_r, max_c = postprocessing_for_bitmap(target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_4, target_map_cand), path_to_cropped_img, '', color_channel=1, particular_channel=0, binarization=True, enhancement=False, resize_times=0, crop_size=image_crop_size)
        
        if stratification_layer == 1:
            postprocessing_for_bitmap(target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_4, target_map_cand), path_to_cropped_img_sup, '_sup_0', color_channel=1, particular_channel=0, binarization=True, enhancement=False, resize_times=0, crop_size=image_crop_size) # 1024
        else:
            postprocessing_for_bitmap(target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_4, target_map_cand), path_to_cropped_img_sup, '_sup_0', color_channel=1, particular_channel=0, binarization=True, enhancement=False, resize_times=0, crop_size=image_crop_size) # 1024
            postprocessing_for_bitmap(target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_4, target_map_cand), path_to_cropped_img_sup, '_sup_1', color_channel=1, particular_channel=0, binarization=True, enhancement=False, resize_times=1, crop_size=image_crop_size) # 512
            postprocessing_for_bitmap(target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_4, target_map_cand), path_to_cropped_img_sup, '_sup_2', color_channel=1, particular_channel=0, binarization=True, enhancement=True, resize_times=2, crop_size=image_crop_size) # 256
        
        #postprocessing_for_bitmap(target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_4, target_map_cand), path_to_cropped_img_sup, '_sup_3', color_channel=1, particular_channel=0, binarization=True, enhancement=True, resize_times=3, crop_size=image_crop_size) # 128
        #postprocessing_for_bitmap(target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_4, target_map_cand), path_to_cropped_img_sup, '_sup_4', color_channel=1, particular_channel=0, binarization=True, enhancement=True, resize_times=4, crop_size=image_crop_size) # 64


        ### gridizing base map into image crops
        info_list = []
        info_list_for_merge = []
        for r in range(0, max_r):
            for c in range(0, max_c):
                for sup in range(0, stratification_layer):
                    this_input_file_name = os.path.join(path_to_cropped_img_sup, target_map_cand.replace('.tif', '')+'_'+str(r)+'_'+str(c)+'_sup_'+str(sup)+'.png')
                    this_output_file_name = this_input_file_name
                    info_list.append([this_input_file_name, this_output_file_name])
                #info_list_for_merge.append([os.path.join(path_to_cropped_img_sup, target_map_cand.replace('.tif', '')+'_'+str(r)+'_'+str(c)+'_sup_0.png'), os.path.join(path_to_cropped_img_sup, target_map_cand.replace('.tif', '')+'_'+str(r)+'_'+str(c)+'_sup_5.png'), 3])

                info_list_for_merge.append([os.path.join(path_to_cropped_img_sup, target_map_cand.replace('.tif', '')+'_'+str(r)+'_'+str(c)+'_sup_0.png'), os.path.join(path_to_cropped_img_sup, target_map_cand.replace('.tif', '')+'_'+str(r)+'_'+str(c)+'_sup_5.png'), stratification_layer])

                #break
            #break
        print(len(info_list), len(info_list_for_merge))
        
        
        ### constructing region map for each image crop
        runningtime_start = datetime.now()
        poly_processed = 0
        with multiprocessing.Pool(10) as pool:
            callback = pool.starmap_async(constructing_region_map.constructing_region_map, [(this_info[0], this_info[1], False, image_crop_size, ) for this_info in info_list])
            multiprocessing_results = callback.get()

            for get_return, this_input in multiprocessing_results:
                if get_return == True:
                    poly_processed += 1
                    #print(str(poly_processed)+'/'+str(len(info_list)), str(datetime.now()-runningtime_start))
                else:
                    print('error', this_input)
        print(target_map_cand, len(info_list), str(datetime.now()-runningtime_start))


        ### merging the region map at different scales
        info_list_v2 = []
        runningtime_start = datetime.now()
        poly_processed = 0
        with multiprocessing.Pool(10) as pool:
            callback = pool.starmap_async(merging_region_map.merging_region_map, [(this_info[0], this_info[1], this_info[2], image_crop_size, ) for this_info in info_list_for_merge])
            multiprocessing_results = callback.get()

            for get_return, this_input in multiprocessing_results:
                if get_return == True:
                    poly_processed += 1
                    #print(str(poly_processed)+'/'+str(len(info_list)), str(datetime.now()-runningtime_start))

                    #info_list_v2.append([os.path.join(path_to_cropped_mask_sup, os.path.basename(this_input).replace('5.png', '0.png')), this_input.replace('.png', '_temp_fill.tif')]) # path to source map crop, path to instance map crop
                    info_list_v2.append([os.path.join(path_to_cropped_mask_sup, os.path.basename(this_input)), this_input.replace('0.png', '6_temp_fill.tif')]) # path to source map crop, path to instance map crop

                    #img_read = cv2.imread(this_input.replace('.png', '_temp_fill.tif'), cv2.IMREAD_UNCHANGED)
                    #print('CV2.IMREAD_UNCHANGED')
                    #print(img_read.shape)
                    #print(np.unique(img_read))
                else:
                    print('error', this_input)
        print(target_map_cand, len(info_list_for_merge), str(datetime.now()-runningtime_start))


        

        if support_training == False:

            item_list_indexed = item_dict_indexed[target_map_cand.replace('.tif', '')]
            item_list_indexed = np.array(item_list_indexed)

            # Extract lists from the dictionary in the order of the keys in the NumPy array
            color_list_indexed_hsv = [color_dict_indexed_hsv[key] for key in item_list_indexed if key in color_dict_indexed_hsv]
            color_list_indexed_hsv = np.array(color_list_indexed_hsv)
            color_list_indexed_rgb = [color_dict_indexed_rgb[key] for key in item_list_indexed if key in color_dict_indexed_rgb]
            color_list_indexed_rgb = np.array(color_list_indexed_rgb)
            color_list_indexed_lab = [color_dict_indexed_lab[key] for key in item_list_indexed if key in color_dict_indexed_lab]
            color_list_indexed_lab = np.array(color_list_indexed_lab)
            #print(item_list_indexed.shape, color_list_indexed.shape)
            
            
            # collect and integrate all the merged results
            img_for_shape = cv2.imread(os.path.join(path_to_target_map_4, target_map_cand))
            img_shape = img_for_shape.shape
            summed_empty_counter = np.zeros((img_shape[0], img_shape[1]), dtype='uint8')
            crop_size = image_crop_size
            


            info_list_v3 = []
            info_list_v4 = []
            runningtime_start = datetime.now()
            poly_processed = 0
            with multiprocessing.Pool(10) as pool:
                callback = pool.starmap_async(assigning_region_map.assigning_region_map, [(this_info[0], this_info[1], color_list_indexed_hsv, color_list_indexed_rgb, color_list_indexed_lab, item_list_indexed, path_to_cropped_intermediate, path_to_cropped_predict, image_crop_size, ) for this_info in info_list_v2])
                multiprocessing_results = callback.get()

                for get_return, this_item_counted, this_split_id, this_r, this_c, integer_masks in multiprocessing_results:
                    if get_return == True:
                        poly_processed += 1
                        if int(this_r) == 0 and int(this_c) == 0: #'0_0' in this_split_id:
                            for this_item in range(0, this_item_counted):
                                info_list_v3.append([item_list_indexed[this_item], this_item])
                                info_list_v4.append([item_list_indexed[this_item], color_list_indexed_hsv[this_item], color_list_indexed_rgb[this_item], color_list_indexed_lab[this_item]])
                        if True:
                            r_0 = int(this_r)*crop_size
                            r_1 = min(int(this_r)*crop_size+crop_size, img_shape[0])
                            c_0 = int(this_c)*crop_size
                            c_1 = min(int(this_c)*crop_size+crop_size, img_shape[1])
                            
                            summed_empty_counter[r_0:r_1, c_0:c_1] = integer_masks[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
                    else:
                        print('error', this_split_id)
            print(target_map_cand, len(info_list_v2), str(datetime.now()-runningtime_start))

                
            print('info_list_v3')
            print(info_list_v3)



            empty_counter = np.zeros((len(info_list_v3), img_shape[0], img_shape[1]), dtype='uint8')
            runningtime_start = datetime.now()
            poly_processed = 0
            with multiprocessing.Pool(10) as pool:
                callback = pool.starmap_async(merging_from_bitmap.merging_from_bitmap_v3, [(this_info[0], this_info[1], path_to_cropped_predict, path_to_merged_predict, img_shape, image_crop_size, ) for this_info in info_list_v3])
                multiprocessing_results = callback.get()

                for get_return, this_item_name, this_item_id in multiprocessing_results: # this_empty_grid
                    if get_return == True:
                        poly_processed += 1
                        #empty_counter[this_item_id] = this_empty_grid
                    else:
                        print('error', this_item_name, this_item_id)
            print(target_map_cand, len(info_list_v3), str(datetime.now()-runningtime_start))
            


            print(summed_empty_counter.shape)
            print(np.unique(summed_empty_counter))
            #cv2.imwrite(os.path.join('Example_Output/PERMIAN_Intermediate/predict/', target_map_cand.replace('.tif', '_count.tif')), summed_empty_counter)


            print('info_list_v4')
            print(info_list_v4)
            print(summed_empty_counter.shape)
            
            
            runningtime_start = datetime.now()
            poly_processed = 0
            with multiprocessing.Pool(10) as pool:
                #callback = pool.starmap_async(thresholding_with_conditions.thresholding_with_conditions, [(os.path.join(path_to_target_map_0, target_map_cand), os.path.join(path_to_target_map_3, target_map_cand), this_info[0], path_to_merged_predict, this_info[1], this_info[2], this_info[3], summed_empty_counter, ) for this_info in info_list_v4])
                callback = pool.starmap_async(thresholding_with_conditions.thresholding_with_conditions, [(os.path.join(path_to_replaced_img_colored, target_map_cand.replace('.tif', '_recolored_v1.png')), os.path.join(path_to_target_map_3, target_map_cand), this_info[0], path_to_merged_predict, this_info[1], this_info[2], this_info[3], summed_empty_counter, ) for this_info in info_list_v4])

                multiprocessing_results = callback.get()

                for get_return, this_item_name in multiprocessing_results:
                    if get_return == True:
                        poly_processed += 1
                    else:
                        print('error', this_item_name)
            print(target_map_cand, len(info_list_v4), str(datetime.now()-runningtime_start))
            print('thresholding_with_conditions')

        else:
            print('...')
    return True



def generating_image_mask(target_map_list, path_to_target_map_3, path_to_replaced_img_colored, path_to_target_groundtruth, path_to_cropped_mask, path_to_cropped_mask_sup, is_training, image_crop_size=1024):
    info_list = []
    for target_map_cand in target_map_list:
        if is_training == True:
            info_list.append([target_map_cand.replace('.tif', ''), os.path.join(path_to_target_groundtruth, target_map_cand.replace('.tif', '_polygon_recoloring.png')), path_to_cropped_mask, '', 3, None, False, False, 0, image_crop_size])
        #info_list.append([target_map_cand.replace('.tif', ''), os.path.join(path_to_target_map_3, target_map_cand.replace('.tif', '.tif_cropping_1_2.tif')), path_to_cropped_mask_sup, '_sup_0', 3, None, False, False, 0, image_crop_size])
        info_list.append([target_map_cand.replace('.tif', ''), os.path.join(path_to_replaced_img_colored, target_map_cand.replace('.tif', '_recolored_v2.png')), path_to_cropped_mask_sup, '_sup_0', 3, None, False, False, 0, image_crop_size])
        info_list.append([target_map_cand.replace('.tif', ''), os.path.join(path_to_replaced_img_colored, target_map_cand.replace('.tif', '_recolored_v1.png')), path_to_cropped_mask_sup, '_sup_1', 3, None, False, False, 0, image_crop_size])
    print(len(info_list))

    runningtime_start = datetime.now()
    map_counter = 0
    with multiprocessing.Pool(10) as pool:
        callback = pool.starmap_async(postprocessing_for_bitmap_worker.postprocessing_for_bitmap, [(this_info[0], this_info[1], this_info[2], this_info[3], this_info[4], this_info[5], this_info[6], this_info[7], this_info[8], this_info[9], ) for this_info in info_list])
        multiprocessing_results = callback.get()

        for max_r, max_c in multiprocessing_results:
            map_counter += 1
    print(len(info_list), str(datetime.now()-runningtime_start))
    return True



def generating_code_mask(target_map_list, path_to_target_map_0, path_to_target_map_5):
    info_list = []
    for target_map_cand in target_map_list:
        info_list.append([os.path.join(path_to_target_map_0, target_map_cand), target_map_cand, path_to_target_map_5])
    print(len(info_list))

    runningtime_start = datetime.now()
    map_counter = 0
    with multiprocessing.Pool(10) as pool:
        callback = pool.starmap_async(getting_color_groundtruth.getting_color_groundtruth, [(this_info[0], this_info[1], this_info[2], ) for this_info in info_list])
        multiprocessing_results = callback.get()

        for get_return in multiprocessing_results:
            if get_return == True:
                map_counter += 1
    print(len(info_list), str(datetime.now()-runningtime_start))

    if os.path.isfile(os.path.join(path_to_target_map_5, 'color_info.csv')) == False:
        with open(os.path.join(path_to_target_map_5, 'color_info.csv'),'w') as fd:
            fd.write('Map_name,Key_name')
            fd.write(',HSV_H_space,HSV_S_space,HSV_V_space,RGB_R_space,RGB_G_space,RGB_B_space,LAB_L_space,LAB_A_space,LAB_B_space,YUV_Y_space,YUV_U_space,YUV_V_space')
            fd.write('\n')
            fd.close()

    for this_info in info_list:
        with open(os.path.join(path_to_target_map_5, 'color_info.csv'),'a') as fd:
            with open(os.path.join(path_to_target_map_5, this_info[1].replace('.tif', '')+'_color_info.csv'),'r') as fdr:
                next(fdr)
                fd.writelines(fdr)
                fdr.close()
            fd.close()
    return True




# Load the image in LAB color space
def split_image(image_path, target_dir_img_small, this_map_name, crop_size=512):
    image = cv2.imread(image_path)

    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    empty_grid = np.zeros((crop_size, crop_size, 3), dtype='uint8')


    for r in range(0,math.ceil(img.shape[0]/crop_size)):
        for c in range(0,math.ceil(img.shape[1]/crop_size)):
            this_output_file_name = os.path.join(target_dir_img_small, str(this_map_name+'_'+str(r)+'_'+str(c)+'.png'))
            
            if (min(r*crop_size+crop_size, img.shape[0]) - r*crop_size <= 0) or (min(c*crop_size+crop_size, img.shape[1]) - c*crop_size <= 0):
                continue
            
            r_0 = r*crop_size
            r_1 = min(r*crop_size+crop_size, img.shape[0])
            c_0 = c*crop_size
            c_1 = min(c*crop_size+crop_size, img.shape[1])

            #print(r, c, r_0, r_1, c_0, c_1)
            if True:
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
            cv2.imwrite(this_output_file_name, img_concat)
    
    return (math.ceil(img.shape[0]/crop_size), math.ceil(img.shape[1]/crop_size))


def merge_recolored(path_to_source_tif, target_dir_img_small_colored, target_dir_img_colored, this_map_name, this_ver='v2', crop_size=256):
    img = cv2.imread(path_to_source_tif)
    #print('merge back to image shape... ', img.shape[0], img.shape[1], ' ...for ', map_name, label_name)
    # original_shape = img.shape
    # print(source_filename, original_shape[0:2])
    empty_grid = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
    empty_flag = True

    for r in range(0,math.ceil(img.shape[0]/crop_size)):
        for c in range(0,math.ceil(img.shape[1]/crop_size)):
            this_block_source = os.path.join(target_dir_img_small_colored, str(this_map_name+'_'+str(r)+'_'+str(c)+'_recolored_'+this_ver+'.png'))
            already_predicted = os.path.isfile(this_block_source)

            if already_predicted == True:
                block_img = cv2.imread(this_block_source)
                block_img = cv2.cvtColor(block_img, cv2.COLOR_BGR2HSV)

                r_0 = r*crop_size
                r_1 = min(r*crop_size+crop_size, img.shape[0])
                c_0 = c*crop_size
                c_1 = min(c*crop_size+crop_size, img.shape[1])
                
                empty_grid[r_0:r_1, c_0:c_1] = block_img[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
            else:
                empty_flag = False
                #break
        #if empty_flag == False:
            #break
    
    #if empty_flag == True:
    if True:
        empty_grid = cv2.cvtColor(empty_grid, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(target_dir_img_colored, this_map_name+'_recolored_'+this_ver+'.png'), empty_grid)
        #print(this_block_source , '>>>', os.path.join(args['output_merged'], str(source_name.split('.')[0]+"_predict.png")))
        #logging.info(f'Merging predicted image {source_name} ...')
        #pbar0.update(1)
    return True



def replacing_color(target_map_list, dir_to_source, path_to_replaced_img, path_to_replaced_img_small, path_to_replaced_img_small_colored, path_to_replaced_img_colored):
    for target_map_cand in target_map_list:
        this_map_name = target_map_cand.replace('.tif', '')
    
        input_image_path = os.path.join(dir_to_source, target_map_cand.replace('.tif', '.tif_cropping_1_2.tif'))
        target_dir_img_small = path_to_replaced_img_small
        target_dir_img_small_colored = path_to_replaced_img_small_colored
        target_dir_img_colored = path_to_replaced_img_colored

        if not os.path.exists(target_dir_img_small):
            os.makedirs(target_dir_img_small)
        if not os.path.exists(target_dir_img_small_colored):
            os.makedirs(target_dir_img_small_colored)
        if not os.path.exists(target_dir_img_colored):
            os.makedirs(target_dir_img_colored)


        this_crop_size = 256
        scale_x, scale_y = split_image(input_image_path, target_dir_img_small, this_map_name, crop_size=this_crop_size)
        this_input_file_name_list = []
        for r in range(0, scale_x):
            for c in range(0, scale_y):
                this_input_file_name = str(this_map_name+'_'+str(r)+'_'+str(c)+'.png')
                this_input_file_name_list.append(this_input_file_name)

        #for this_info in this_input_file_name_list:
        #    color_replacement(target_dir_img_small, target_dir_img_small_colored, this_info)
        this_input_file_name_processed_list = []
        with multiprocessing.Pool(int(PROCESSES)) as pool:
            callback = pool.starmap_async(color_replacement.color_replacement, [(target_dir_img_small, target_dir_img_small_colored, this_info ) for this_info in this_input_file_name_list])
            multiprocessing_results = callback.get()
            
            for this_image_name, flag in multiprocessing_results:
                this_input_file_name_processed_list.append(this_image_name)
        print(len(this_input_file_name_processed_list))

        merge_recolored(input_image_path, target_dir_img_small_colored, target_dir_img_colored, this_map_name, this_ver='v1', crop_size=this_crop_size)
        merge_recolored(input_image_path, target_dir_img_small_colored, target_dir_img_colored, this_map_name, this_ver='v2', crop_size=this_crop_size)
    return True



def storing_raster_output(target_map_list, path_to_target_map_0, path_to_merged_predict, dir_to_raster_output):
    for target_map_cand in target_map_list:
        with open(os.path.join(path_to_target_map_0, target_map_cand.replace('.tif', '.json'))) as f:
            gj = json.load(f)

        poly_counter = 0
        legend_name = []
        legend_name_check = []

        for this_gj in gj['shapes']:
            names = this_gj['label']
            features = this_gj['points']
            
            if '_poly' not in names:
                continue
            if names in legend_name_check:
                #print(target_map_cand, names)
                continue
            legend_name_check.append(names)
            legend_name.append(names)
            

            path_to_extracted_raster_output = os.path.join(path_to_merged_predict, target_map_cand.replace('.tif', '')+'_'+names+'_predict.png')
            if os.path.isfile(path_to_extracted_raster_output) == False:
                print(path_to_extracted_raster_output, ' ... file not found...')
            else:
                path_to_extraction_tif = os.path.join(dir_to_raster_output, target_map_cand.replace('.tif', '')+'_'+names+'_predict.tif')
                shutil.copyfile(path_to_extracted_raster_output, path_to_extraction_tif)

            poly_counter = poly_counter+1
        print(poly_counter)
    return True



def main(dir_to_intermediate, path_to_source_map, path_to_source_groundtruth, dir_to_raster_output, map_name, path_to_tif, path_to_json):
    training = False
    support_training = False
    image_crop_size = 512 # 1024


    for this_category in range(0, 1):
        overall_runningtime_start = datetime.now()
        
        #path_to_source_map = 'J:/Research/Data/'+str(category[this_category])+'/'
        #path_to_source_groundtruth = 'J:/Research/Data/'+str(category[this_category])+'_groundtruth/'
        #dir_to_intermediate = 'Example_Output'
        path_to_source_groundtruth = None

        path_to_target_map_0 = os.path.join(dir_to_intermediate, 'PERMIAN_General/', str(category[this_category])+'_source/')
        path_to_target_groundtruth = os.path.join(dir_to_intermediate, 'PERMIAN_General/', str(category[this_category])+'_groundtruth/')
        

        path_to_target_map_1 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_grayscale/')
        path_to_target_map_2 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_binary/')
        path_to_target_map_3 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_roi/')
        path_to_target_map_4 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'/')
        path_to_target_map_5 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_info/')
        path_to_target_map_6 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_text/')
        path_to_target_map_7 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_text_intermediate/')
        path_to_target_map_8 = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_item/')
        path_to_cropped_img = os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'imgs')
        path_to_cropped_img_sup = os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'imgs', 'sup')
        path_to_cropped_mask = os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'masks')
        path_to_cropped_mask_sup = os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'data', 'cma_small', 'masks', 'sup')
        path_to_cropped_predict = os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'predict', 'cma_small')
        path_to_cropped_intermediate = os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'predict', 'cma_small_colored')
        path_to_merged_predict = os.path.join(dir_to_intermediate, 'PERMIAN_Intermediate', 'predict', 'cma')
        path_to_replaced_img = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/')
        path_to_replaced_img_small = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/', 'small')
        path_to_replaced_img_small_colored = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/', 'small_colored')
        path_to_replaced_img_colored = os.path.join(dir_to_intermediate, 'PERMIAN_General', str(category[this_category])+'_replace/', 'colored')

        directory_setting(category, dir_to_intermediate)
        print('Step(0/9) directory_setting done... ')
        target_map_list = identifying_target_map(this_category, path_to_source_map, path_to_target_map_0, dir_to_intermediate, map_name, path_to_tif, path_to_json)
        print('Step(1/9) identifying_target_map done... ')


        cropping_map_area(target_map_list, path_to_source_map, path_to_target_map_3)
        print('Step(2/9) cropping_map_area done... ')
        generating_color_groundtruth(target_map_list, path_to_source_map, path_to_source_groundtruth, path_to_target_map_1, path_to_target_map_2, path_to_target_map_3, path_to_target_map_4, path_to_target_groundtruth, training)
        print('Step(3/9) generating_color_groundtruth done... ')
        cropping_legend_area(target_map_list, path_to_target_map_0, path_to_target_map_2, path_to_target_map_8)
        print('Step(4/9) cropping_legend_area done... ')
        
        replacing_color(target_map_list, path_to_target_map_3, path_to_replaced_img, path_to_replaced_img_small, path_to_replaced_img_small_colored, path_to_replaced_img_colored)
        print('Step(5/9) replacing_color done... ')
        
        generating_image_mask(target_map_list, path_to_target_map_3, path_to_replaced_img_colored, path_to_target_groundtruth, path_to_cropped_mask, path_to_cropped_mask_sup, training, image_crop_size)
        print('Step(6/9) generating_image_mask done... ')
        generating_code_mask(target_map_list, path_to_target_map_0, path_to_target_map_5)
        print('Step(7/9) generating_code_mask done... ')

        generating_region_map(target_map_list, path_to_target_map_0, path_to_replaced_img_colored, path_to_target_map_3, path_to_target_map_4, path_to_target_map_5, path_to_cropped_img, path_to_cropped_img_sup, path_to_cropped_mask_sup, path_to_cropped_intermediate, path_to_cropped_predict, path_to_merged_predict, support_training, image_crop_size)
        print('Step(8/9) generating_region_map done... ')

        storing_raster_output(target_map_list, path_to_target_map_0, path_to_merged_predict, dir_to_raster_output)
        print('Step(9/9) storing_raster_output done... ')


        print('Running_time checkpoint... ')
        print(datetime.now()-overall_runningtime_start)
    
    return True



def run_permian(dir_to_intermediate, path_to_tif, path_to_json, path_to_source_groundtruth, input_dir_to_raster_output, input_thread=12, input_efficiency_trade_off=1):
    # Split the file path into directory and file components
    path_to_source_map, map_name = os.path.split(path_to_tif)

    main(dir_to_intermediate, path_to_source_map, path_to_source_groundtruth, input_dir_to_raster_output, map_name, path_to_tif, path_to_json)
    return True



