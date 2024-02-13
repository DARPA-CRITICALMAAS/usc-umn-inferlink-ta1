import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from scipy import ndimage
from collections import Counter




def processing_uncharted_json_batch(input_legend_segmentation, path_to_tif, target_map_name, output_segmentation):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    target_id = -1
    for this_gj in gj['images']:
        this_id = this_gj['id']
        this_file_name = this_gj['file_name']
        #print(target_map_name.split('.')[0], this_file_name)
        if target_map_name.split('.')[0] in this_file_name.replace('%20', ' '):
            target_id = this_id
            legend_area_placeholder = np.zeros((this_gj['height'], this_gj['width']), dtype='uint8')
            break
    
    if target_id == -1:
        return False

    for this_gj in gj['annotations']:
        if this_gj['image_id'] == target_id:
            if this_gj['category_id'] == 2:
                coord_counter = 0
                poly_coord = []
                this_coord = []
                for value in this_gj['segmentation'][0]:
                    this_coord.append(int(value))
                    if coord_counter % 2 == 1:
                        this_coord = np.array(this_coord)
                        poly_coord.append(this_coord)
                        this_coord = []
                    coord_counter += 1
                poly_coord = np.array(poly_coord)

                cv2.fillConvexPoly(legend_area_placeholder, poly_coord, 1)
                legend_area_placeholder[legend_area_placeholder > 0] = 255
                cv2.imwrite(output_segmentation, legend_area_placeholder)

                print_bgr = cv2.imread(path_to_tif)
                print_bgr = cv2.cvtColor(print_bgr, cv2.COLOR_BGR2RGB)
                print_bgr = cv2.bitwise_and(print_bgr, print_bgr, mask=legend_area_placeholder)
                print_bgr = cv2.cvtColor(print_bgr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_segmentation.replace('_expected_crop_region.tif', '_crop.tif'), print_bgr)
                break

    return True



def processing_uncharted_json_single(input_legend_segmentation, input_image, target_map_name, output_segmentation):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    img0 = cv2.imread(input_image)
    gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    legend_area_placeholder = np.zeros((gray0.shape[0],gray0.shape[1]), dtype='uint8')

    for this_gj in gj['segments']:
        if 'map' in this_gj['class_label']:
            cv2.fillConvexPoly(legend_area_placeholder, np.array(this_gj['poly_bounds']), 1)
            legend_area_placeholder[legend_area_placeholder > 0] = 255

    cv2.imwrite(output_segmentation, legend_area_placeholder)

    print_bgr = cv2.imread(input_image)
    print_bgr = cv2.cvtColor(print_bgr, cv2.COLOR_BGR2RGB)
    print_bgr = cv2.bitwise_and(print_bgr, print_bgr, mask=legend_area_placeholder)
    print_bgr = cv2.cvtColor(print_bgr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_segmentation.replace('_expected_crop_region.tif', '_crop.tif'), print_bgr)

    return True
                    







def preprocessing_coworker(map_id, path_to_bound, file_name, path_to_tif, path_to_json, solutiona_dir, crop_legend):
    
    filename=file_name.replace('.json', '.tif')
    print('Working on map:', file_name)
    file_path=path_to_tif
    test_json=path_to_json
    
    img0 = cv2.imread(file_path)
    hsv0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
    rgb0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)


    lower_black = np.array([0,0,0])
    upper_black = np.array([130,130,130]) #100

    blank = np.zeros((img0.shape[0],img0.shape[1],img0.shape[2]),dtype=np.uint8)
    blank[0:img0.shape[0],0:img0.shape[1],0:img0.shape[2]]=255

    # create a mask to only preserve current legend color in the basemap
    mask_box = cv2.inRange(rgb0, lower_black, upper_black)
    res_box = cv2.bitwise_and(blank,blank, mask=mask_box)

    # convert to grayscale 
    detected_gray0 = cv2.cvtColor(res_box, cv2.COLOR_BGR2GRAY)
    img_bw0 = cv2.threshold(detected_gray0, 60, 255, cv2.THRESH_BINARY)[1] # 127

    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_blackregion.png'
    cv2.imwrite(out_file_path0, img_bw0)



    lower_black1 = np.array([100,100,100])
    upper_black1 = np.array([180,180,180])

    # create a mask to only preserve current legend color in the basemap
    mask_box1 = cv2.inRange(rgb0, lower_black1, upper_black1)
    res_box1 = cv2.bitwise_and(blank,blank, mask=mask_box1)

    # convert to grayscale 
    detected_gray1 = cv2.cvtColor(res_box1, cv2.COLOR_BGR2GRAY)
    img_bw01 = cv2.threshold(detected_gray1, 60, 255, cv2.THRESH_BINARY)[1] # 127

    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_grayregion.png'
    cv2.imwrite(out_file_path0, img_bw01)



    blank = np.zeros((img0.shape[0],img0.shape[1],img0.shape[2]),dtype=np.uint8)
    blank[0:img0.shape[0],0:img0.shape[1],0:img0.shape[2]] = 255



    try:
        processing_uncharted_json_batch(path_to_bound, path_to_tif, file_name.replace('.json', '.tif'), solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
    except:
        processing_uncharted_json_single(path_to_bound, path_to_tif, file_name.replace('.json', '.tif'), solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
    

    #img_bw0_arg = np.argwhere(img_bw0 == 255)
    #print(img_bw0_arg.shape)
    crop_rgb2 = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop.tif')
    crop_rgb2 = cv2.cvtColor(crop_rgb2, cv2.COLOR_BGR2RGB)

    selected_map_for_examination = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')
    selected_map_for_examination = cv2.cvtColor(selected_map_for_examination, cv2.COLOR_BGR2GRAY)


    crop_rgb1 = np.copy(crop_rgb2)
    crop_img_bw0 = cv2.bitwise_and(img_bw0,img_bw0, mask=selected_map_for_examination)
    crop_img_bw01 = cv2.bitwise_and(img_bw01,img_bw01, mask=selected_map_for_examination)

    #print(crop_rgb1.shape)
    #plt.imshow(crop_rgb1)

    print_bgr = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2BGR)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop.png'
    cv2.imwrite(out_file_path0, print_bgr)
    #break

    img_bw0_arg = np.argwhere(crop_img_bw0 == 255)
    #print(img_bw0_arg.shape)

    
    #plt.imshow(crop_img_bw01)
    #plt.show()
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_grayregion.png'
    cv2.imwrite(out_file_path0, crop_img_bw01)

    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_crop_blackregion.png'
    cv2.imwrite(out_file_path0, crop_img_bw0)




    trans_hsv1 = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2HSV)

    print_bgr = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2BGR)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png'
    cv2.imwrite(out_file_path0, print_bgr)


    # without mean shift (for rgb)
    trans_rgb0 = cv2.cvtColor(trans_hsv1, cv2.COLOR_HSV2RGB)
    trans_hsv0 = cv2.cvtColor(trans_rgb0, cv2.COLOR_RGB2HSV)
    #crop_rgb1 = np.copy(trans_rgb0)
    #trans_hsv1 = np.copy(trans_hsv0)
    
    # mean shift (for rgb and hsv)
    spatialRadius = 10
    colorRadius = 25
    maxPyramidLevel = 2

    img = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_rgb_part = cv2.bitwise_and(img_rgb, img_rgb, mask=selected_map_for_examination)
    

    result = cv2.pyrMeanShiftFiltering(img_rgb, spatialRadius, colorRadius, maxPyramidLevel)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    #crop_rgb1 = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    #trans_hsv1 = cv2.cvtColor(crop_rgb1, cv2.COLOR_RGB2HSV)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black_mean_shift.png'
    cv2.imwrite(out_file_path0, result)


    '''Text removal from boundaries / Begin'''
    # black_pixel_for_text
    # black_text_for_text
    # black_line_for_text // black_line_for_text = black_pixel_for_text - black_text_for_text
    img_rb = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_remove_black.png')
    img_bound = cv2.imread(solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_expected_crop_region.tif')

    img_background_v0 = np.copy(img_rb)
    img_background_v0 = cv2.cvtColor(img_background_v0, cv2.COLOR_RGB2GRAY)

    lower_black_text = np.array([0])
    upper_black_text = np.array([70])
    mask_box_text0 = cv2.inRange(img_background_v0, lower_black_text, upper_black_text)
    res_box_text1 = cv2.bitwise_and(img_bound, img_bound, mask=mask_box_text0)

    threshold_text = cv2.medianBlur(res_box_text1,3)
    threshold_text_strict = cv2.medianBlur(res_box_text1,5)

    threshold_text_strict = ndimage.gaussian_filter(threshold_text_strict, sigma=20)
    '''
    out_file_path0=solutiona_dir+'intermediate6/0_poly_t1g.png'
    cv2.imwrite(out_file_path0, threshold_text_strict)
    '''
    threshold_text_strict[threshold_text_strict > 255.0*0.05] = 255
    threshold_text_strict[threshold_text_strict <= 255.0*0.05] = 0

    threshold_text0 = cv2.bitwise_and(threshold_text, threshold_text_strict)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_text_detection.png'
    cv2.imwrite(out_file_path0, threshold_text0)

    threshold_line = cv2.subtract(threshold_text, threshold_text0)
    kernel_dilate = np.ones((5, 5), np.uint8)
    threshold_line = cv2.dilate(threshold_line, kernel_dilate, iterations=1)
    out_file_path0=solutiona_dir+'intermediate7/'+file_name.replace('.json', '')+'_boundary_detection.png'
    cv2.imwrite(out_file_path0, threshold_line)

    '''Text removal from boundaries / End'''
    
    return selected_map_for_examination