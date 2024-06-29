import cv2
import numpy as np
import os
import json


def detect_color_boundaries(img_hsv):
    # Threshold black and gray pixels
    # Define ranges for black and gray in HSV space
    black_lower, black_upper = np.array([0, 0, 0]), np.array([180, 255, 50])
    gray_lower, gray_upper = np.array([0, 0, 51]), np.array([180, 50, 220])
    
    # Create masks for black and gray
    black_mask = cv2.inRange(img_hsv, black_lower, black_upper)
    gray_mask = cv2.inRange(img_hsv, gray_lower, gray_upper)
    
    # Detect boundaries in H channel using Canny
    edges = cv2.Canny(img_hsv[:,:,0], 100, 200)  # Use the H channel
    
    # Buffer black and gray areas and apply inverse mask
    kernel = np.ones((5,5), np.uint8)
    black_dilated = cv2.dilate(black_mask, kernel, iterations=1)
    gray_dilated = cv2.dilate(gray_mask, kernel, iterations=1)
    
    # Combine black and gray masks and create an inverse mask
    combined_mask = cv2.bitwise_or(black_dilated, gray_dilated)
    inverse_combined_mask = cv2.bitwise_not(combined_mask)
    
    # Apply inverse mask to edges
    final_edges = cv2.bitwise_and(edges, edges, mask=inverse_combined_mask)
    
    # Output the binary image indicating boundaries
    return final_edges


def color2gray(source, output_1, output_2, input_3, output_4):
    # load image into lab space
    img = cv2.imread(source)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    #print(img_lab.shape)

    # write image in l space
    # grayscale image, focusing on black and gray colors
    cv2.imwrite(output_1, img_lab[:, :, 0])

    # binary image
    th, im_th = cv2.threshold(img_lab[:, :, 0], 100, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_2, im_th)

    # overlapping with region of interest, binary image
    bound = cv2.imread(input_3)
    bound = cv2.cvtColor(bound, cv2.COLOR_BGR2GRAY)
    bound_th = cv2.bitwise_and(im_th, bound)
    cv2.imwrite(output_4, bound_th)


    ##### modify39
    # find boundaries among colors, excluding black and gray colors
    # // only applied for feature extraction, not for colorization
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_boundary = detect_color_boundaries(img_hsv)
    #cv2.imwrite(output_2.replace('.tif', '_color_boundary.tif'), color_boundary)

    bound_th = cv2.bitwise_or(255-bound_th, color_boundary)
    bound_th = 255-bound_th
    cv2.imwrite(output_4, bound_th)
    #cv2.imwrite(output_4.replace('.tif', '_color_boundary.tif'), bound_th)
    


def colorization_groundtruth(source, source_map_name, dir_to_groundtruth, dir_to_target):
    # generate the colored image based on the groundtruth polygon for each map key
    #print('Working on map:', source_map_name)
    file_name = source.replace('.tif', '.json')
    test_json = file_name
    file_path = source
    
    #print(test_json)
    img000 = cv2.imread(file_path)

    with open(test_json) as f:
        gj = json.load(f)
    json_height = gj['imageHeight']
    json_width = gj['imageWidth']
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
        black_threshold = 30 #130
        white_threshold = 250 #245

        lower_black_rgb_trimmed0 = np.array([0,0,0])
        upper_black_rgb_trimmed0 = np.array([130,130,130])
        mask_test_img_legend = cv2.inRange(img_legend, lower_black_rgb_trimmed0, upper_black_rgb_trimmed0)
        if np.sum(mask_test_img_legend == 255) > np.sum(img_legend > 0) * 0.25:
            black_threshold = 30
        
        rgb_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
        hsv_trimmed = np.zeros((img_legend.shape[2], img_legend.shape[0], img_legend.shape[1]), dtype='uint8')
        rgb_trimmed = rgb_trimmed.astype(float)
        hsv_trimmed = hsv_trimmed.astype(float)
        for dimension in range(0, 3):
            rgb_trimmed[dimension] = np.copy(img_legend[:,:,dimension]).astype(float)
            hsv_trimmed[dimension] = np.copy(hsv_legend[:,:,dimension]).astype(float)

        rgb_trimmed_temp = np.copy(rgb_trimmed)
        rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan

        rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        hsv_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        hsv_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        hsv_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan



        if np.sum(np.isnan(hsv_trimmed)) >= (hsv_trimmed.shape[0]*hsv_trimmed.shape[1]*hsv_trimmed.shape[2]):
            color_avg_holder = np.array((0,0,0), dtype='uint8')
        else:
            color_avg_holder = np.array([int(np.nanquantile(rgb_trimmed[0],.5)),int(np.nanquantile(rgb_trimmed[1],.5)),int(np.nanquantile(rgb_trimmed[2],.5))])
        color_avg.append(color_avg_holder)



    # list the path to all the groundtruth polygons
    candidate_file_path = []
    for this_poly in range(0, poly_counter):
        if os.path.isfile(os.path.join(dir_to_groundtruth, (map_name+'_'+legend_name[this_poly]+'_poly.tif'))) == True:
            candidate_file_path.append(os.path.join(dir_to_groundtruth, (map_name+'_'+legend_name[this_poly]+'_poly.tif')))
    print('Working on map: ' + map_name + ' (with legends: ' + str(len(candidate_file_path)) + ')')
    if len(candidate_file_path) == 0:
        print('no groundtruth provided...')
    

    base_canvas = np.zeros((img000.shape[0], img000.shape[1], 3), dtype=np.uint8)
    legend_counting = 0

    #for candidate_polygon_groundtruth in os.listdir(data_boundary_dir):
    for this_poly in range(0, len(candidate_file_path)):
        candidate_polygon_groundtruth = candidate_file_path[this_poly]
        if '.tif' in candidate_polygon_groundtruth:
            if map_name in candidate_polygon_groundtruth: #[0: len(file_name)+1]:
                legend_counting = legend_counting + 1
                #this_candidate_groundtruth = os.path.join(data_boundary_dir, candidate_polygon_groundtruth)
                this_candidate_groundtruth = candidate_polygon_groundtruth

                candidate_canvas = np.full((img000.shape[0], img000.shape[1], 3), color_avg[this_poly], dtype=np.uint8)

                this_candidate = cv2.imread(this_candidate_groundtruth)
                this_candidate = cv2.cvtColor(this_candidate, cv2.COLOR_BGR2GRAY)

                candidate_canvas = cv2.bitwise_and(candidate_canvas, candidate_canvas, mask=this_candidate)
                base_canvas = cv2.add(base_canvas, candidate_canvas)

    base_canvas = cv2.cvtColor(base_canvas, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(dir_to_target, map_name+'_polygon_recoloring.png'), base_canvas)


def generating_colorization_groundtruth(target_map_cand, path_to_source_map, path_to_target_map_1, path_to_target_map_2, path_to_source_groundtruth, path_to_target_groundtruth, path_to_target_map_3, path_to_target_map_4, is_training):
    try:
        color2gray(os.path.join(path_to_source_map, target_map_cand), os.path.join(path_to_target_map_1, target_map_cand), os.path.join(path_to_target_map_2, target_map_cand), os.path.join(path_to_target_map_3, target_map_cand), os.path.join(path_to_target_map_4, target_map_cand))
        if is_training == True:
            colorization_groundtruth(os.path.join(path_to_source_map, target_map_cand), target_map_cand, path_to_source_groundtruth, path_to_target_groundtruth)
    except:
        return False, target_map_cand
    return True, target_map_cand