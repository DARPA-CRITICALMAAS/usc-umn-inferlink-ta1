import cv2
import numpy as np
import os
import json


def getting_color_groundtruth(source, source_map_name, dir_to_info):
    # generate the colored image based on the groundtruth polygon for each map key
    #print('Working on map:', source_map_name)
    file_name = source.replace('.tif', '.json')
    test_json = file_name
    file_path = source
    
    #print(test_json)
    img000 = cv2.imread(file_path)

    with open(test_json) as f:
        gj = json.load(f)
    rescale_factor_0 = 1.0
    rescale_factor_1 = 1.0

    poly_counter = 0
    color_avg = []
    map_name = source_map_name.replace('.tif', '')
    legend_name = []
    legend_name_check = []


    for this_gj in gj['shapes']:
        names = this_gj['label']
        features = this_gj['points']
        
        if '_poly' not in names:
            continue
        if names in legend_name_check:
            continue
        legend_name_check.append(names)
        legend_name.append(names.replace('_poly',''))
        poly_counter = poly_counter+1


        ### Read json source for the legend
        geoms = np.array(features)
        y_min = int(np.min(geoms, axis=0)[0]*rescale_factor_1)
        y_max = int(np.max(geoms, axis=0)[0]*rescale_factor_1)
        x_min = int(np.min(geoms, axis=0)[1]*rescale_factor_0)
        x_max = int(np.max(geoms, axis=0)[1]*rescale_factor_0)

        img_legend = np.zeros((x_max-x_min, y_max-y_min, 3), dtype='uint8')
        img_legend = np.copy(img000[x_min:x_max, y_min:y_max, :])
        
        img_legend = cv2.cvtColor(img_legend, cv2.COLOR_BGR2RGB)
        img_legend = img_legend[int(img_legend.shape[0]/8):int(img_legend.shape[0]*7/8), int(img_legend.shape[1]/8):int(img_legend.shape[1]*7/8), :]
        hsv_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2HSV)
        lab_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2LAB)
        yuv_legend = cv2.cvtColor(img_legend, cv2.COLOR_RGB2YUV)

        black_threshold = 30 #130
        white_threshold = 250 #245

        lower_black_rgb_trimmed0 = np.array([0,0,0])
        upper_black_rgb_trimmed0 = np.array([130,130,130])
        mask_test_img_legend = cv2.inRange(img_legend, lower_black_rgb_trimmed0, upper_black_rgb_trimmed0)
        if np.sum(mask_test_img_legend == 255) > np.sum(img_legend > 0) * 0.25:
            black_threshold = 30
        
        rgb_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
        hsv_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
        lab_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
        yuv_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
        rgb_trimmed = rgb_trimmed.astype(float)
        hsv_trimmed = hsv_trimmed.astype(float)
        lab_trimmed = lab_trimmed.astype(float)
        yuv_trimmed = yuv_trimmed.astype(float)
        for dimension in range(0, 3):
            rgb_trimmed[dimension] = np.copy(img_legend[:,:,dimension]).astype(float)
            hsv_trimmed[dimension] = np.copy(hsv_legend[:,:,dimension]).astype(float)
            lab_trimmed[dimension] = np.copy(lab_legend[:,:,dimension]).astype(float)
            yuv_trimmed[dimension] = np.copy(yuv_legend[:,:,dimension]).astype(float)

        rgb_trimmed_temp = np.copy(rgb_trimmed)
        lab_trimmed_temp = np.copy(lab_trimmed)
        yuv_trimmed_temp = np.copy(yuv_trimmed)
        rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        lab_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        yuv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        lab_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        yuv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        lab_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        yuv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan

        rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        lab_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        yuv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        lab_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        yuv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        lab_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        yuv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan


        if np.sum(np.isnan(hsv_trimmed)) >= (hsv_trimmed.shape[0]*hsv_trimmed.shape[1]*hsv_trimmed.shape[2]):
            color_avg_holder = np.array((0.0,0.0,0.0), dtype=np.float16)
            color_avg_holder_rgb = np.array([int(np.nanquantile(rgb_trimmed_temp[0],.5))/255.0,int(np.nanquantile(rgb_trimmed_temp[1],.5))/255.0,int(np.nanquantile(rgb_trimmed_temp[2],.5))/255.0])
            color_avg_holder_lab = np.array([int(np.nanquantile(lab_trimmed_temp[0],.5))/255.0,int(np.nanquantile(lab_trimmed_temp[1],.5))/255.0,int(np.nanquantile(lab_trimmed_temp[2],.5))/255.0])
            color_avg_holder_yuv = np.array([int(np.nanquantile(yuv_trimmed_temp[0],.5))/255.0,int(np.nanquantile(yuv_trimmed_temp[1],.5))/255.0,int(np.nanquantile(yuv_trimmed_temp[2],.5))/255.0])
        else:
            color_avg_holder = np.array([int(np.nanquantile(hsv_trimmed[0],.5))/179.0,int(np.nanquantile(hsv_trimmed[1],.5))/255.0,int(np.nanquantile(hsv_trimmed[2],.5))/255.0])
            color_avg_holder_rgb = np.array([int(np.nanquantile(rgb_trimmed[0],.5))/255.0,int(np.nanquantile(rgb_trimmed[1],.5))/255.0,int(np.nanquantile(rgb_trimmed[2],.5))/255.0])
            color_avg_holder_lab = np.array([int(np.nanquantile(lab_trimmed[0],.5))/255.0,int(np.nanquantile(lab_trimmed[1],.5))/255.0,int(np.nanquantile(lab_trimmed[2],.5))/255.0])
            color_avg_holder_yuv = np.array([int(np.nanquantile(yuv_trimmed[0],.5))/255.0,int(np.nanquantile(yuv_trimmed[1],.5))/255.0,int(np.nanquantile(yuv_trimmed[2],.5))/255.0])

        #color_avg.append(color_avg_holder)


        concat_name = str(map_name)+'_'+str(names)
        if os.path.isfile(os.path.join(dir_to_info, map_name+'_color_info.csv')) == False:
            with open(os.path.join(dir_to_info, map_name+'_color_info.csv'),'w') as fd:
                fd.write('Map_name,Key_name')
                fd.write(',HSV_H_space,HSV_S_space,HSV_V_space,RGB_R_space,RGB_G_space,RGB_B_space,LAB_L_space,LAB_A_space,LAB_B_space,YUV_Y_space,YUV_U_space,YUV_V_space')
                fd.write('\n')
                fd.close()
        with open(os.path.join(dir_to_info, map_name+'_color_info.csv'),'a') as fd:
            fd.write(map_name+','+concat_name)
            fd.write(','+str(color_avg_holder[0])+','+str(color_avg_holder[1])+','+str(color_avg_holder[2]))
            fd.write(','+str(color_avg_holder_rgb[0])+','+str(color_avg_holder_rgb[1])+','+str(color_avg_holder_rgb[2]))
            fd.write(','+str(color_avg_holder_lab[0])+','+str(color_avg_holder_lab[1])+','+str(color_avg_holder_lab[2]))
            fd.write(','+str(color_avg_holder_yuv[0])+','+str(color_avg_holder_yuv[1])+','+str(color_avg_holder_yuv[2]))
            fd.write('\n')
            fd.close()
    
    return True