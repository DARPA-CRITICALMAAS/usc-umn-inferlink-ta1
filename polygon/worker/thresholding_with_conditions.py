import math
import os
import cv2
import numpy as np
from scipy.ndimage import label, binary_erosion
from numpy import exp
from scipy.sparse import coo_matrix


def refine_and_nucleation(mask_source, instance_source, summed_counter_source, thresholding_lower=0.2, thresholding_upper_1=0.2, thresholding_upper_2=0.3, resizing=True):
    mask = np.copy(mask_source)
    instance = np.copy(instance_source)
    summed_counter = np.copy(summed_counter_source)

    if resizing:
        size_source = mask.shape
        mask = cv2.resize(mask, (size_source[1]//2, size_source[0]//2), interpolation=cv2.INTER_NEAREST)
        instance = cv2.resize(instance, (size_source[1]//2, size_source[0]//2), interpolation=cv2.INTER_NEAREST)
        summed_counter = cv2.resize(summed_counter, (size_source[1]//2, size_source[0]//2), interpolation=cv2.INTER_NEAREST)

    # Ensure mask is a binary numpy array (0 and 1 only)
    mask = (mask > 0).astype(np.uint8)

    # Ensure instance is a binary numpy array (0 and 1 only)
    instance = (instance > 0).astype(np.uint8)

    # Label connected components in the instance
    component_map, num_features = label(instance)
    if num_features == 0:
        empty_map = np.zeros_like(mask)
        return empty_map

    # Calculate overlap of each component with the mask
    total_pixels = np.bincount(component_map.ravel(), minlength=num_features+1)[1:]  # +1 to handle zero index
    masked_pixels = np.bincount(component_map.ravel(), weights=mask.ravel(), minlength=num_features+1)[1:]
    masked_summed = np.bincount(component_map.ravel(), weights=summed_counter.ravel(), minlength=num_features+1)[1:]

    # Calculate the number of unique mask labels within each component
    mask_labeled, num_mask_components = label(mask)

    # Create row indices (component labels) and column indices (mask labels)
    rows = component_map.ravel()
    cols = mask_labeled.ravel()

    # Filter out background (label 0 in both component and mask)
    valid = (rows > 0) & (cols > 0)
    rows = rows[valid]-1 # Adjust index to start at 0
    cols = cols[valid]-1 # Adjust index to start at 0

    # Create a sparse matrix where entries are counts of each (row, col) pair
    data = np.ones_like(rows)
    matrix = coo_matrix((data, (rows, cols)), shape=(num_features , num_mask_components ))
    #print(matrix.shape)

    # Convert to CSR format for row-based operations
    matrix_csr = matrix.tocsr()

    # Sum across columns to count the number of unique masks per component
    component_counts = np.diff(matrix_csr.indptr) - 1  # Subtract one to ignore the zeroth label
    #print('component_counts', component_counts.shape)
    #print(component_counts)

    # Calculate boundaries within each component
    boundaries = mask - binary_erosion(mask)
    boundary_counts = np.bincount(component_map.ravel(), weights=boundaries.ravel(), minlength=num_features+1)[1:]
    #print('boundary_counts', boundary_counts.shape)
    
    # Calculate overlap ratio for all components at once
    overlap_ratio = masked_pixels / total_pixels

    # Calculate summed overlap ratio for all components
    summed_ratio = masked_summed / total_pixels # how many current extraction overlap on this instance
    #print('overlap_ratio', overlap_ratio.shape)

    
    epsilon = 1e-6  # To prevent division by zero
    threshold_upper = (summed_ratio / (component_counts + epsilon)) * exp((-masked_pixels + boundary_counts) / (total_pixels + epsilon))
    threshold_lower = summed_ratio / (component_counts + epsilon)

    # Linear transformations for summed_ratio
    adjusted_ratio_upper = thresholding_upper_1 + thresholding_upper_2 * np.clip(threshold_upper, 0, 1.0)  # adjust range to 0.2 to 0.5
    adjusted_ratio_lower = 0.1 + thresholding_lower * np.clip(threshold_lower, 0, 1.0)  # adjust range to 0.1 to 0.3
    
    '''
    adjusted_ratio_upper = 0.5
    adjusted_ratio_lower = 0.2
    '''

    # Find components based on adjusted thresholds
    valid_components = np.where(overlap_ratio > adjusted_ratio_upper)[0] + 1
    invalid_components = np.where(overlap_ratio < adjusted_ratio_lower)[0] + 1

    # Create a binary map of these valid and invalid components respectively
    valid_map = np.isin(component_map, valid_components) # nucleate all instances recognized as valid components
    invalid_map = np.isin(component_map, invalid_components) # dissolve all instance recognized as invalid components

    # store the result of positive instances
    updated_masked_instance = cv2.bitwise_or(valid_map.astype(np.uint8), mask)

    # store the result of negative instances
    updated_masked_instance = cv2.bitwise_and(updated_masked_instance, 1-invalid_map.astype(np.uint8))

    if resizing:
        updated_masked_instance = cv2.resize(updated_masked_instance, (size_source[1], size_source[0]), interpolation=cv2.INTER_NEAREST)

    return updated_masked_instance





def identifying_peak_id(arr):
    if arr.shape[0] <= 1:
        return 0
    
    # sort the array and determine the smaller half
    filtered_arr = arr[arr != 0]
    sorted_filtered_arr = np.sort(filtered_arr)
    half_size = len(sorted_filtered_arr) // 2
    smaller_half = sorted_filtered_arr[:half_size]
    #smaller_half = sorted_filtered_arr[half_size:]
    
    # calculate mean and standard deviation of the smaller half
    mean_smaller_half = np.mean(smaller_half)
    std_dev_smaller_half = np.std(smaller_half)
    
    # calculate the target peak-thresholding/-identification value (mean + 2 standard deviation)
    target_value = mean_smaller_half + std_dev_smaller_half*2 #2
    #print(target_value)
    
    
    # Try to find the first index with a value larger than the target value
    above_target_indices = np.where(arr > target_value)[0]
    if above_target_indices.size > 0:
        return max(1, above_target_indices[0]-1)
    
    # If no element found above target value, find the first index with a value above the median
    median_value = max(np.median(arr), np.mean(arr)) # np.median(arr)
    return max(1, np.argmax(arr > median_value)-1)
    

    '''
    # Try to find the second index with a value larger than the target value
    above_target_indices = np.where(arr > target_value)[0]
    if above_target_indices.size > 1:
        return max(1, above_target_indices[1]-1)
    
    # If fewer than two elements are found above target value, find the second index with a value above the median
    #median_value = max(np.median(arr), np.mean(arr))
    median_value = max(np.median(smaller_half), np.mean(smaller_half))
    above_median_indices = np.where(arr > median_value)[0]
    if above_median_indices.size > 1:
        return max(1, above_median_indices[1]-1)
    
    # Try to find the first index with a value larger than the target value
    if above_target_indices.size > 0:
        return max(1, above_target_indices[0]-1)
    
    # If no element found above target value, or fewer than two elements are found above median, find the first index with a value above the median
    #median_value = max(np.median(arr), np.mean(arr))
    return max(1, np.argmax(arr > median_value)-1)
    '''



def thresholding_with_conditions(path_to_source_image, path_to_source_roi, this_item_name, path_to_merged_predict, target_hsv, target_rgb, target_lab, summed_counter):
    NUCLEATION = True


    source_roi = cv2.imread(path_to_source_roi)
    source_roi = cv2.cvtColor(source_roi, cv2.COLOR_BGR2GRAY)
    source_roi[source_roi>0] = 255

    #print(path_to_source_image)
    #path_to_source_image = 'Example_Output/PERMIAN_General/testing_groundtruth/CO_DenverW_polygon_recoloring.png'
    source_image = cv2.imread(path_to_source_image)
    source_image_hsv = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
    source_image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    source_image_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    print('source_image_hsv', source_image_hsv.shape)

    # load instance-based extraction with relaxed hsv, rgb, and lab color spaces
    # mask source image in hsv color space for further nucleation
    path_to_pre_thresholding = os.path.join(path_to_merged_predict, this_item_name+'_predict.png')
    image_pre_thresholding = cv2.imread(path_to_pre_thresholding)
    image_pre_thresholding = cv2.cvtColor(image_pre_thresholding, cv2.COLOR_BGR2GRAY)
    image_pre_thresholding[image_pre_thresholding>0] = 255

    # relax the buffer area of the identified instances
    path_to_pre_thresholding_relaxed = os.path.join(path_to_merged_predict, this_item_name+'_predict_r2_instance.png')
    image_pre_thresholding_relaxed = cv2.imread(path_to_pre_thresholding_relaxed)
    image_pre_thresholding_relaxed = cv2.cvtColor(image_pre_thresholding_relaxed, cv2.COLOR_BGR2GRAY)
    image_pre_thresholding_relaxed[image_pre_thresholding_relaxed>0] = 255

    print('image_pre_thresholding', image_pre_thresholding.shape, type(image_pre_thresholding))

    instance_highlighted_image_hsv = cv2.bitwise_and(source_image_hsv, source_image_hsv, mask=image_pre_thresholding)
    instance_highlighted_image_hsv_relaxed = cv2.bitwise_and(source_image_hsv, source_image_hsv, mask=image_pre_thresholding_relaxed)
    instance_highlighted_image_hsv_relaxed_roi = cv2.bitwise_and(source_image_hsv, source_image_hsv, mask=source_roi)
    instance_highlighted_image_rgb = cv2.bitwise_and(source_image_rgb, source_image_rgb, mask=image_pre_thresholding)
    instance_highlighted_image_rgb_relaxed = cv2.bitwise_and(source_image_rgb, source_image_rgb, mask=image_pre_thresholding_relaxed)
    instance_highlighted_image_rgb_relaxed_roi = cv2.bitwise_and(source_image_rgb, source_image_rgb, mask=source_roi)
    instance_highlighted_image_lab = cv2.bitwise_and(source_image_lab, source_image_lab, mask=image_pre_thresholding)
    instance_highlighted_image_lab_relaxed = cv2.bitwise_and(source_image_lab, source_image_lab, mask=image_pre_thresholding_relaxed)
    instance_highlighted_image_lab_relaxed_roi = cv2.bitwise_and(source_image_lab, source_image_lab, mask=source_roi)


    this_thresholding_median_hsv = target_hsv #[this_item_id, :]
    this_thresholding_median_rgb = target_rgb
    this_thresholding_median_lab = target_lab


    kernel_erode = np.ones((8,8), np.uint8) # TODO further enlarge
    kernel_dilate = np.ones((3,3), np.uint8)

    this_must_include = np.where(summed_counter == 1, 255, 0)
    print('this_must_include', this_must_include.shape, type(this_must_include))
    this_must_include = cv2.bitwise_and(this_must_include.astype(np.uint8), image_pre_thresholding.astype(np.uint8))

    #return True, this_item_name

    # keep record of valid-candidate pixels and thresholded pixels
    pre_thresholded_ratio = np.mean(image_pre_thresholding)
    if pre_thresholded_ratio > 0:

        ########## HSV
        relaxed_ratio = []
        relaxed_output = {}
        selected_relaxation = -1
        previous_ratio = 0.0
        # hyperparameter...
        for adaptive_relaxation in range(0, 6): # 5
            # apply relaxed threshold in hsv color space from legend item
            this_thresholding_range_hsv = []

            # hyperparameter...
            if adaptive_relaxation == -1:
                this_thresholding_range_hsv.append([max(0, this_thresholding_median_hsv[0]), max(0, this_thresholding_median_hsv[1]), max(0, this_thresholding_median_hsv[2])])
                this_thresholding_range_hsv.append([min(180, this_thresholding_median_hsv[0]), min(255, this_thresholding_median_hsv[1]), min(255, this_thresholding_median_hsv[2])])
            else:
                this_thresholding_range_hsv.append([max(0, this_thresholding_median_hsv[0]-3*(adaptive_relaxation+1)), max(0, this_thresholding_median_hsv[1]-10-2*(adaptive_relaxation)), max(0, this_thresholding_median_hsv[2]-10-2*(adaptive_relaxation))])
                this_thresholding_range_hsv.append([min(180, this_thresholding_median_hsv[0]+3*(adaptive_relaxation+1)), min(255, this_thresholding_median_hsv[1]+10+2*(adaptive_relaxation)), min(255, this_thresholding_median_hsv[2]+10+2*(adaptive_relaxation))])

            this_thresholding_range_hsv = np.array(this_thresholding_range_hsv)
            raster_masked_image_hsv = cv2.inRange(instance_highlighted_image_hsv, this_thresholding_range_hsv[0], this_thresholding_range_hsv[1])

            # erode to remove noisy pixels
            raster_masked_image_hsv = cv2.erode(raster_masked_image_hsv, kernel_erode, iterations=1)

            # dilate to expand within instance-based mask
            
            for iterative_dilation in range(0, 3):
                raster_masked_image_hsv = cv2.dilate(raster_masked_image_hsv, kernel_dilate, iterations=1)
                raster_masked_image_hsv = cv2.bitwise_and(raster_masked_image_hsv.astype(np.uint8), image_pre_thresholding.astype(np.uint8))

            # overlap with this_must_include (pixels those only belong to one item)
            raster_masked_image_hsv = cv2.bitwise_or(raster_masked_image_hsv, this_must_include)

            relaxed_output[str(adaptive_relaxation)] = raster_masked_image_hsv

            cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_hsv_relaxation_'+str(adaptive_relaxation)+'.png'), raster_masked_image_hsv)

            this_thresholded_ratio = np.mean(raster_masked_image_hsv)
            #relaxed_ratio.append(this_thresholded_ratio)

            # default to select the first-round relaxation with a non-zero thresholded ratio
            if previous_ratio == 0 and this_thresholded_ratio > 0:
                selected_relaxation = adaptive_relaxation
                relaxed_ratio.append(1.0)
            

            # if the change in ratio is larger than 1.5, abort and take the previous non-zero round relaxation output
            # hyperparameter...
            if previous_ratio > 0 and this_thresholded_ratio/previous_ratio > 2.5: # this_thresholded_ratio/previous_ratio is always larger than or equal to 1 # 5.0
                selected_relaxation = adaptive_relaxation-1
                relaxed_ratio.append(this_thresholded_ratio/previous_ratio)
                break


            previous_ratio = this_thresholded_ratio
        
        
        if selected_relaxation == -1:
            # if all thresholded output return zero pixels...
            selected_relaxation = 0
        else:
            # ideal to select the relaxation before a large gap in change of ratio (to maximize possible coverage)
            relaxed_ratio = np.array(relaxed_ratio)
            selected_relaxation = identifying_peak_id(relaxed_ratio)

        # retrieve the thresholded output
        retrieved_relaxation_hsv = relaxed_output[str(selected_relaxation)]
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_hsv_relaxation_x.png'), retrieved_relaxation_hsv)
        # release memory
        #relaxed_output = {}


        ### relaxation
        '''
        # apply the same thresholding level to the relaxed area...
        if selected_relaxation >= 0:
            # apply relaxed threshold in hsv color space from legend item
            this_thresholding_range_hsv = []

            this_thresholding_range_hsv.append([max(0, this_thresholding_median_hsv[0]-3*(selected_relaxation+1)), max(0, this_thresholding_median_hsv[1]-10-2*(selected_relaxation)), max(0, this_thresholding_median_hsv[2]-10-2*(selected_relaxation))])
            this_thresholding_range_hsv.append([min(180, this_thresholding_median_hsv[0]+3*(selected_relaxation+1)), min(255, this_thresholding_median_hsv[1]+10+2*(selected_relaxation)), min(255, this_thresholding_median_hsv[2]+10+2*(selected_relaxation))])
            this_thresholding_range_hsv = np.array(this_thresholding_range_hsv)

            retrieved_relaxation_hsv_relaxed = cv2.inRange(instance_highlighted_image_hsv_relaxed, this_thresholding_range_hsv[0], this_thresholding_range_hsv[1])
            retrieved_relaxation_hsv_relaxed_roi = cv2.inRange(instance_highlighted_image_hsv_relaxed_roi, this_thresholding_range_hsv[0], this_thresholding_range_hsv[1])
            
            # erode to remove noisy pixels
            retrieved_relaxation_hsv_relaxed = cv2.erode(retrieved_relaxation_hsv_relaxed, kernel_erode, iterations=1)
            retrieved_relaxation_hsv_relaxed_roi = cv2.erode(retrieved_relaxation_hsv_relaxed_roi, kernel_erode, iterations=1)

            # overlap with this_must_include (pixels those only belong to one item)
            #retrieved_relaxation_hsv_relaxed = cv2.bitwise_or(retrieved_relaxation_hsv_relaxed, this_must_include)

            # dilate to expand within instance-based mask
            for iterative_dilation in range(0, 3):
                retrieved_relaxation_hsv_relaxed = cv2.dilate(retrieved_relaxation_hsv_relaxed, kernel_dilate, iterations=1)
                retrieved_relaxation_hsv_relaxed = cv2.bitwise_and(retrieved_relaxation_hsv_relaxed.astype(np.uint8), image_pre_thresholding_relaxed.astype(np.uint8))
                retrieved_relaxation_hsv_relaxed_roi = cv2.dilate(retrieved_relaxation_hsv_relaxed_roi, kernel_dilate, iterations=1)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_hsv_relaxation_x2.png'), retrieved_relaxation_hsv_relaxed)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_hsv_relaxation_x3.png'), retrieved_relaxation_hsv_relaxed_roi)
        '''









        ########## RGB
        relaxed_ratio = []
        relaxed_output = {}
        selected_relaxation = -1
        previous_ratio = 0.0
        # hyperparameter...
        for adaptive_relaxation in range(0, 6): # 5
            # apply relaxed threshold in hsv color space from legend item
            this_thresholding_range_rgb = []

            # hyperparameter...
            if adaptive_relaxation == -1:
                this_thresholding_range_rgb.append([max(0, this_thresholding_median_rgb[0]), max(0, this_thresholding_median_rgb[1]), max(0, this_thresholding_median_rgb[2])])
                this_thresholding_range_rgb.append([min(255, this_thresholding_median_rgb[0]), min(255, this_thresholding_median_rgb[1]), min(255, this_thresholding_median_rgb[2])])
            else:
                this_thresholding_range_rgb.append([max(0, this_thresholding_median_rgb[0]-2*(adaptive_relaxation)), max(0, this_thresholding_median_rgb[1]-2*(adaptive_relaxation)), max(0, this_thresholding_median_rgb[2]-2*(adaptive_relaxation))])
                this_thresholding_range_rgb.append([min(255, this_thresholding_median_rgb[0]+2*(adaptive_relaxation)), min(255, this_thresholding_median_rgb[1]+2*(adaptive_relaxation)), min(255, this_thresholding_median_rgb[2]+2*(adaptive_relaxation))])

            this_thresholding_range_rgb = np.array(this_thresholding_range_rgb)
            raster_masked_image_rgb = cv2.inRange(instance_highlighted_image_rgb, this_thresholding_range_rgb[0], this_thresholding_range_rgb[1])

            # erode to remove noisy pixels
            raster_masked_image_rgb = cv2.erode(raster_masked_image_rgb, kernel_erode, iterations=1)

            # dilate to expand within instance-based mask
            
            for iterative_dilation in range(0, 3):
                raster_masked_image_rgb = cv2.dilate(raster_masked_image_rgb, kernel_dilate, iterations=1)
                raster_masked_image_rgb = cv2.bitwise_and(raster_masked_image_rgb.astype(np.uint8), image_pre_thresholding.astype(np.uint8))

            # overlap with this_must_include (pixels those only belong to one item)
            raster_masked_image_rgb = cv2.bitwise_or(raster_masked_image_rgb, this_must_include)

            relaxed_output[str(adaptive_relaxation)] = raster_masked_image_rgb

            cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_rgb_relaxation_'+str(adaptive_relaxation)+'.png'), raster_masked_image_rgb)

            this_thresholded_ratio = np.mean(raster_masked_image_rgb)
            #relaxed_ratio.append(this_thresholded_ratio)

            # default to select the first-round relaxation with a non-zero thresholded ratio
            if previous_ratio == 0 and this_thresholded_ratio > 0:
                selected_relaxation = adaptive_relaxation
                relaxed_ratio.append(1.0)
            

            # if the change in ratio is larger than 1.5, abort and take the previous non-zero round relaxation output
            # hyperparameter...
            if previous_ratio > 0 and this_thresholded_ratio/previous_ratio > 2.5: # this_thresholded_ratio/previous_ratio is always larger than or equal to 1 # 5.0
                selected_relaxation = adaptive_relaxation-1
                relaxed_ratio.append(this_thresholded_ratio/previous_ratio)
                break


            previous_ratio = this_thresholded_ratio
        
        
        if selected_relaxation == -1:
            # if all thresholded output return zero pixels...
            selected_relaxation = 0
        else:
            # ideal to select the relaxation before a large gap in change of ratio (to maximize possible coverage)
            relaxed_ratio = np.array(relaxed_ratio)
            selected_relaxation = identifying_peak_id(relaxed_ratio)

        # retrieve the thresholded output
        retrieved_relaxation_rgb = relaxed_output[str(selected_relaxation)]
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_rgb_relaxation_x.png'), retrieved_relaxation_rgb)
        # release memory
        #relaxed_output = {}


        ### relaxation
        '''
        # apply the same thresholding level to the relaxed area...
        if selected_relaxation >= 0:
            # apply relaxed threshold in hsv color space from legend item
            this_thresholding_range_rgb = []

            this_thresholding_range_rgb.append([max(0, this_thresholding_median_rgb[0]-2*(adaptive_relaxation)), max(0, this_thresholding_median_rgb[1]-2*(adaptive_relaxation)), max(0, this_thresholding_median_rgb[2]-2*(adaptive_relaxation))])
            this_thresholding_range_rgb.append([min(255, this_thresholding_median_rgb[0]+2*(adaptive_relaxation)), min(255, this_thresholding_median_rgb[1]+2*(adaptive_relaxation)), min(255, this_thresholding_median_rgb[2]+2*(adaptive_relaxation))])
            this_thresholding_range_rgb = np.array(this_thresholding_range_rgb)

            retrieved_relaxation_rgb_relaxed = cv2.inRange(instance_highlighted_image_rgb_relaxed, this_thresholding_range_rgb[0], this_thresholding_range_rgb[1])
            retrieved_relaxation_rgb_relaxed_roi = cv2.inRange(instance_highlighted_image_rgb_relaxed_roi, this_thresholding_range_rgb[0], this_thresholding_range_rgb[1])
            
            # erode to remove noisy pixels
            retrieved_relaxation_rgb_relaxed = cv2.erode(retrieved_relaxation_rgb_relaxed, kernel_erode, iterations=1)
            retrieved_relaxation_rgb_relaxed_roi = cv2.erode(retrieved_relaxation_rgb_relaxed_roi, kernel_erode, iterations=1)

            # overlap with this_must_include (pixels those only belong to one item)
            #retrieved_relaxation_rgb_relaxed = cv2.bitwise_or(retrieved_relaxation_rgb_relaxed, this_must_include)

            # dilate to expand within instance-based mask
            for iterative_dilation in range(0, 3):
                retrieved_relaxation_rgb_relaxed = cv2.dilate(retrieved_relaxation_rgb_relaxed, kernel_dilate, iterations=1)
                retrieved_relaxation_rgb_relaxed = cv2.bitwise_and(retrieved_relaxation_rgb_relaxed.astype(np.uint8), image_pre_thresholding_relaxed.astype(np.uint8))
                retrieved_relaxation_rgb_relaxed_roi = cv2.dilate(retrieved_relaxation_rgb_relaxed_roi, kernel_dilate, iterations=1)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_rgb_relaxation_x2.png'), retrieved_relaxation_rgb_relaxed)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_rgb_relaxation_x3.png'), retrieved_relaxation_rgb_relaxed_roi)
        '''




        ########## LAB
        relaxed_ratio = []
        relaxed_output = {}
        selected_relaxation = -1
        previous_ratio = 0.0
        # hyperparameter...
        for adaptive_relaxation in range(0, 6): # 5
            # apply relaxed threshold in hsv color space from legend item
            this_thresholding_range_lab = []

            # hyperparameter...
            if adaptive_relaxation == -1:
                this_thresholding_range_lab.append([max(0, this_thresholding_median_lab[0]), max(0, this_thresholding_median_lab[1]), max(0, this_thresholding_median_lab[2])])
                this_thresholding_range_lab.append([min(255, this_thresholding_median_lab[0]), min(255, this_thresholding_median_lab[1]), min(255, this_thresholding_median_lab[2])])
            else:
                this_thresholding_range_lab.append([max(0, this_thresholding_median_lab[0]-4*(adaptive_relaxation)), max(0, this_thresholding_median_lab[1]-2*(adaptive_relaxation)), max(0, this_thresholding_median_lab[2]-2*(adaptive_relaxation))])
                this_thresholding_range_lab.append([min(255, this_thresholding_median_lab[0]+4*(adaptive_relaxation)), min(255, this_thresholding_median_lab[1]+2*(adaptive_relaxation)), min(255, this_thresholding_median_lab[2]+2*(adaptive_relaxation))])

            this_thresholding_range_lab = np.array(this_thresholding_range_lab)
            raster_masked_image_lab = cv2.inRange(instance_highlighted_image_lab, this_thresholding_range_lab[0], this_thresholding_range_lab[1])

            # erode to remove noisy pixels
            raster_masked_image_lab = cv2.erode(raster_masked_image_lab, kernel_erode, iterations=1)

            # dilate to expand within instance-based mask
            
            for iterative_dilation in range(0, 3):
                raster_masked_image_lab = cv2.dilate(raster_masked_image_lab, kernel_dilate, iterations=1)
                raster_masked_image_lab = cv2.bitwise_and(raster_masked_image_lab.astype(np.uint8), image_pre_thresholding.astype(np.uint8))

            # overlap with this_must_include (pixels those only belong to one item)
            raster_masked_image_lab = cv2.bitwise_or(raster_masked_image_lab, this_must_include)

            relaxed_output[str(adaptive_relaxation)] = raster_masked_image_lab

            cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_lab_relaxation_'+str(adaptive_relaxation)+'.png'), raster_masked_image_lab)

            this_thresholded_ratio = np.mean(raster_masked_image_lab)
            #relaxed_ratio.append(this_thresholded_ratio)

            # default to select the first-round relaxation with a non-zero thresholded ratio
            if previous_ratio == 0 and this_thresholded_ratio > 0:
                selected_relaxation = adaptive_relaxation
                relaxed_ratio.append(1.0)
            

            # if the change in ratio is larger than 1.5, abort and take the previous non-zero round relaxation output
            # hyperparameter...
            if previous_ratio > 0 and this_thresholded_ratio/previous_ratio > 2.5: # this_thresholded_ratio/previous_ratio is always larger than or equal to 1 # 5.0
                selected_relaxation = adaptive_relaxation-1
                relaxed_ratio.append(this_thresholded_ratio/previous_ratio)
                break


            previous_ratio = this_thresholded_ratio
        
        
        if selected_relaxation == -1:
            # if all thresholded output return zero pixels...
            selected_relaxation = 0
        else:
            # ideal to select the relaxation before a large gap in change of ratio (to maximize possible coverage)
            relaxed_ratio = np.array(relaxed_ratio)
            selected_relaxation = identifying_peak_id(relaxed_ratio)

        # retrieve the thresholded output
        retrieved_relaxation_lab = relaxed_output[str(selected_relaxation)]
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_lab_relaxation_x.png'), retrieved_relaxation_lab)
        # release memory
        #relaxed_output = {}


        ### relaxation
        '''
        # apply the same thresholding level to the relaxed area...
        if selected_relaxation >= 0:
            # apply relaxed threshold in hsv color space from legend item
            this_thresholding_range_lab = []

            this_thresholding_range_lab.append([max(0, this_thresholding_median_lab[0]-4*(adaptive_relaxation)), max(0, this_thresholding_median_lab[1]-2*(adaptive_relaxation)), max(0, this_thresholding_median_lab[2]-2*(adaptive_relaxation))])
            this_thresholding_range_lab.append([min(255, this_thresholding_median_lab[0]+4*(adaptive_relaxation)), min(255, this_thresholding_median_lab[1]+2*(adaptive_relaxation)), min(255, this_thresholding_median_lab[2]+2*(adaptive_relaxation))])
            this_thresholding_range_lab = np.array(this_thresholding_range_lab)

            retrieved_relaxation_lab_relaxed = cv2.inRange(instance_highlighted_image_lab_relaxed, this_thresholding_range_lab[0], this_thresholding_range_lab[1])
            retrieved_relaxation_lab_relaxed_roi = cv2.inRange(instance_highlighted_image_lab_relaxed_roi, this_thresholding_range_lab[0], this_thresholding_range_lab[1])
            
            # erode to remove noisy pixels
            retrieved_relaxation_lab_relaxed = cv2.erode(retrieved_relaxation_lab_relaxed, kernel_erode, iterations=1)
            retrieved_relaxation_lab_relaxed_roi = cv2.erode(retrieved_relaxation_lab_relaxed_roi, kernel_erode, iterations=1)

            # overlap with this_must_include (pixels those only belong to one item)
            #retrieved_relaxation_lab_relaxed = cv2.bitwise_or(retrieved_relaxation_lab_relaxed, this_must_include)

            # dilate to expand within instance-based mask
            for iterative_dilation in range(0, 3):
                retrieved_relaxation_lab_relaxed = cv2.dilate(retrieved_relaxation_lab_relaxed, kernel_dilate, iterations=1)
                retrieved_relaxation_lab_relaxed = cv2.bitwise_and(retrieved_relaxation_lab_relaxed.astype(np.uint8), image_pre_thresholding_relaxed.astype(np.uint8))
                retrieved_relaxation_lab_relaxed_roi = cv2.dilate(retrieved_relaxation_lab_relaxed_roi, kernel_dilate, iterations=1)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_lab_relaxation_x2.png'), retrieved_relaxation_lab_relaxed)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_lab_relaxation_x3.png'), retrieved_relaxation_lab_relaxed_roi)
        '''




        '''
        retrieved_relaxation = cv2.bitwise_or(retrieved_relaxation_hsv, retrieved_relaxation_rgb)
        retrieved_relaxation = cv2.bitwise_or(retrieved_relaxation, retrieved_relaxation_lab)

        #retrieved_relaxation_r2 = cv2.bitwise_or(retrieved_relaxation_hsv_relaxed, retrieved_relaxation_rgb_relaxed)
        #retrieved_relaxation_r2 = cv2.bitwise_or(retrieved_relaxation_r2, retrieved_relaxation_lab_relaxed)

        #retrieved_relaxation_r3 = cv2.bitwise_or(retrieved_relaxation_hsv_relaxed_roi, retrieved_relaxation_rgb_relaxed_roi)
        #retrieved_relaxation_r3 = cv2.bitwise_or(retrieved_relaxation_r3, retrieved_relaxation_lab_relaxed_roi
        
        

        
        # start nucleation
        updated_masked_instance = refine_and_nucleation(retrieved_relaxation, image_pre_thresholding, summed_counter, thresholding_lower=0.2, thresholding_upper_1=0.5, thresholding_upper_2=0.2)
        #updated_masked_instance_r2 = refine_and_nucleation(retrieved_relaxation_r2, image_pre_thresholding_relaxed, summed_counter, thresholding_lower=0.2, thresholding_upper_1=0.5, thresholding_upper_2=0.2)

        # some post-processing with must-included region
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_preliminary_v1.png'), updated_masked_instance)
        #cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_preliminary_v2.png'), updated_masked_instance_r2)
        #cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_preliminary_v3.png'), retrieved_relaxation_r3)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_relaxation_base.png'), this_must_include)
        #updated_masked_instance = cv2.bitwise_or(updated_masked_instance.astype(np.uint8), updated_masked_instance_r2.astype(np.uint8))
        #updated_masked_instance = cv2.bitwise_or(updated_masked_instance.astype(np.uint8), retrieved_relaxation_r3.astype(np.uint8))
        updated_masked_instance = cv2.bitwise_or(updated_masked_instance.astype(np.uint8), this_must_include.astype(np.uint8))
        '''



        retrieved_relaxation = cv2.bitwise_or(retrieved_relaxation_hsv, retrieved_relaxation_rgb)
        retrieved_relaxation = cv2.bitwise_or(retrieved_relaxation, retrieved_relaxation_lab)

        
        if NUCLEATION == True:
            # start nucleation
            updated_masked_instance = refine_and_nucleation(retrieved_relaxation, image_pre_thresholding, summed_counter, thresholding_lower=0.2, thresholding_upper_1=0.5, thresholding_upper_2=0.2)

        # some post-processing with must-included region
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_preliminary_v0.png'), retrieved_relaxation)
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_relaxation_base.png'), this_must_include)

        if NUCLEATION == True:
            cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_preliminary_v1.png'), updated_masked_instance)
            updated_masked_instance = cv2.bitwise_or(updated_masked_instance.astype(np.uint8), this_must_include.astype(np.uint8))
        else:
            print('Skip nucleation at this version due to concerns regarding posssible memory-allocation issue...')
            updated_masked_instance = cv2.bitwise_or(retrieved_relaxation.astype(np.uint8), this_must_include.astype(np.uint8))
        

        # add a dilation to handle boundaries
        #kernel_dilate = np.ones((2,2), np.uint8)
        #updated_masked_instance = cv2.dilate(updated_masked_instance, kernel_dilate, iterations=1)

    else:
        updated_masked_instance = image_pre_thresholding
    
    if np.mean(updated_masked_instance) < 0.00001:
        print('Roll back to original thresholded output...')
        updated_masked_instance = image_pre_thresholding

    updated_masked_instance[updated_masked_instance > 0] = 255
    #cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_nucleation.png'), updated_masked_instance)
    cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_permian.png'), updated_masked_instance)
    return True, this_item_name

