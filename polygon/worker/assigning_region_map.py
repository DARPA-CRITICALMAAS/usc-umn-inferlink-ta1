import cv2
import os
import numpy as np




def apply_mask_to_base(base_image, mask_condition):
    """Applies a given masking condition to an HSV/ RGB/ LAB image, returning the masked image."""
    return base_image[:, mask_condition]

def median_base_of_masked_image(base_masked):
    """Calculates the median of an HSV/ RGB/ LAB image only if it has valid pixels."""
    if base_masked.size > 0:
        return np.median(base_masked, axis=1)
    return None

def mask_conditions_hsv(hsv_image, instance_mask):
    """Generates mask conditions for black and gray pixels within a specific instance area."""
    threshold_for_black = 30  # V less than this is considered black
    threshold_for_gray_s = 40  # S less than this is considered gray
    threshold_for_gray_v = 50  # V less than this is considered gray

    # Mask for black pixels
    mask_black = (hsv_image[2, :] > threshold_for_black) & instance_mask
    # Mask for black and gray pixels
    #mask_gray = ((hsv_image[1, :] > threshold_for_gray_s) & (hsv_image[2, :] > threshold_for_gray_v)) & instance_mask

    #return mask_black, mask_gray
    return mask_black


def mask_conditions_rgb(rgb_image, instance_mask):
    """Generates mask conditions for black and gray pixels within a specific instance area."""
    threshold_for_black = 50  # RGB less than this is considered black
    gray_tolerance = 10

    # Mask for black pixels
    is_black = (rgb_image[0, :] <= threshold_for_black) & (rgb_image[1, :] <= threshold_for_black) & (rgb_image[2, :] <= threshold_for_black)
    mask_black = ~is_black & instance_mask
    
    # Mask for black and gray pixels
    diff_rg = np.abs(rgb_image[0, :, :] - rgb_image[1, :, :])
    diff_rb = np.abs(rgb_image[0, :, :] - rgb_image[2, :, :])
    diff_gb = np.abs(rgb_image[1, :, :] - rgb_image[2, :, :])
    is_gray = (diff_rg <= gray_tolerance) & (diff_rb <= gray_tolerance) & (diff_gb <= gray_tolerance)

    # Create the final mask: not gray and in instance_mask
    mask_gray = ~is_gray & instance_mask
    mask_gray = mask_gray & mask_black

    return mask_black, mask_gray


def mask_conditions_lab(lab_image, instance_mask):
    """Generates mask conditions for black and gray pixels within a specific instance area."""
    threshold_for_black = 50  # L less than this is considered black
    gray_tolerance = 10

    # Mask for black pixels
    mask_black = (lab_image[0, :] > threshold_for_black) & instance_mask
    
    # Mask for black and gray pixels
    non_gray_a = np.abs(lab_image[1, :] - 128) > gray_tolerance
    non_gray_b = np.abs(lab_image[2, :] - 128) > gray_tolerance

    # Create the final mask: not gray and in instance_mask
    mask_gray = (non_gray_a | non_gray_b) & instance_mask
    mask_gray = mask_gray & mask_black

    return mask_black, mask_gray


def calculate_median_hsv(instance_map, path_to_source_image, this_item_name, path_to_cropped_intermediate):
    source_map = cv2.imread(path_to_source_image)
    hsv_image = cv2.cvtColor(source_map, cv2.COLOR_BGR2HSV)
    rgb_image = cv2.cvtColor(source_map, cv2.COLOR_BGR2RGB)
    lab_image = cv2.cvtColor(source_map, cv2.COLOR_BGR2LAB)

    hsv_image = np.transpose(hsv_image, (2, 0, 1))
    rgb_image = np.transpose(rgb_image, (2, 0, 1))
    lab_image = np.transpose(lab_image, (2, 0, 1))

    # Calculate median HSV values for each unique instance with masking conditions
    unique_instances = np.unique(instance_map)
    instance_medians_hsv = {}
    instance_medians_rgb = {}
    instance_medians_lab = {}

    # initialize a dictionary to support mapping between instance_id and numpy_id; should not be a problem now, but in case
    instance_to_npid = {}
    npid_to_instance = {}
    npid_counting = 0

    # initialize a numpy array for visualization
    bool_for_visualization = True
    if bool_for_visualization == True:
        output_hsv_image = np.zeros_like(hsv_image)  # Create an output image with the same size and type
        output_rgb_image = np.zeros_like(rgb_image)  # Create an output image with the same size and type
        output_lab_image = np.zeros_like(lab_image)  # Create an output image with the same size and type

    for instance_id in unique_instances:
        instance_mask = (instance_map == instance_id)
        
        if np.any(instance_mask):
            # HSV color space

            # Get mask conditions within the instance area
            #mask_black, mask_gray = mask_conditions(hsv_image, instance_mask)
            hsv_mask_black = mask_conditions_hsv(hsv_image, instance_mask)
            
            # Mask the image with mask_b, mask_a, and base instance mask respectively
            #hsv_masked_b = apply_mask_to_base(hsv_image, mask_gray)
            hsv_masked_a = apply_mask_to_base(hsv_image, hsv_mask_black)
            hsv_masked_base = apply_mask_to_base(hsv_image, instance_mask)
            
            # Calculate medians
            #median_hsv_b = median_base_of_masked_image(hsv_masked_b)
            median_hsv_a = median_base_of_masked_image(hsv_masked_a)
            median_hsv_base = median_base_of_masked_image(hsv_masked_base)
            
            # Decide which median to use based on availability of valid pixels
            #if median_hsv_b is not None:
            #    instance_medians[instance_id] = median_hsv_b
            if median_hsv_a is not None:
                instance_medians_hsv[instance_id] = median_hsv_a
            else:
                instance_medians_hsv[instance_id] = median_hsv_base if median_hsv_base is not None else np.array([0, 0, 0])
            

            # RGB color space
            rgb_mask_black, rgb_mask_gray = mask_conditions_rgb(rgb_image, instance_mask)
            
            rgb_masked_b = apply_mask_to_base(rgb_image, rgb_mask_gray)
            rgb_masked_a = apply_mask_to_base(rgb_image, rgb_mask_black)
            rgb_masked_base = apply_mask_to_base(rgb_image, instance_mask)
            
            median_rgb_b = median_base_of_masked_image(rgb_masked_b)
            median_rgb_a = median_base_of_masked_image(rgb_masked_a)
            median_rgb_base = median_base_of_masked_image(rgb_masked_base)
            
            if median_rgb_b is not None:
                instance_medians_rgb[instance_id] = median_rgb_b
            elif median_rgb_a is not None:
                instance_medians_rgb[instance_id] = median_rgb_a
            else:
                instance_medians_rgb[instance_id] = median_rgb_base if median_rgb_base is not None else np.array([0, 0, 0])


            
            # LAB color space
            lab_mask_black, lab_mask_gray = mask_conditions_lab(lab_image, instance_mask)
            
            lab_masked_b = apply_mask_to_base(lab_image, lab_mask_gray)
            lab_masked_a = apply_mask_to_base(lab_image, lab_mask_black)
            lab_masked_base = apply_mask_to_base(lab_image, instance_mask)
            
            median_lab_b = median_base_of_masked_image(lab_masked_b)
            median_lab_a = median_base_of_masked_image(lab_masked_a)
            median_lab_base = median_base_of_masked_image(lab_masked_base)
            
            if median_lab_b is not None:
                instance_medians_lab[instance_id] = median_lab_b
            elif median_lab_a is not None:
                instance_medians_lab[instance_id] = median_lab_a
            else:
                instance_medians_lab[instance_id] = median_lab_base if median_lab_base is not None else np.array([0, 128, 128])

            

        else:
            instance_medians_hsv[instance_id] = np.array([0, 0, 0])  # Placeholder for instances without valid pixels
            instance_medians_rgb[instance_id] = np.array([0, 0, 0])  # Placeholder for instances without valid pixels
            instance_medians_lab[instance_id] = np.array([0, 128, 128])  # Placeholder for instances without valid pixels
        

        if bool_for_visualization == True:
            for i in range(3):  # There are 3 channels: H, S, V
                output_hsv_image[i,:,:][instance_mask] = instance_medians_hsv[instance_id][i]
                output_rgb_image[i,:,:][instance_mask] = instance_medians_rgb[instance_id][i]
                output_lab_image[i,:,:][instance_mask] = instance_medians_lab[instance_id][i]

        instance_to_npid[instance_id] = npid_counting
        npid_to_instance[npid_counting] = instance_id
        npid_counting += 1
    
    if bool_for_visualization == True:
        output_hsv_image = np.transpose(output_hsv_image, (1, 2, 0))
        output_hsv_image = cv2.cvtColor(output_hsv_image, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(path_to_cropped_intermediate, this_item_name+'_colored_hsv.png'), output_hsv_image)

        output_rgb_image = np.transpose(output_rgb_image, (1, 2, 0))
        output_rgb_image = cv2.cvtColor(output_rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(path_to_cropped_intermediate, this_item_name+'_colored_rgb.png'), output_rgb_image)

        output_lab_image = np.transpose(output_lab_image, (1, 2, 0))
        output_lab_image = cv2.cvtColor(output_lab_image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(path_to_cropped_intermediate, this_item_name+'_colored_lab.png'), output_lab_image)


    return instance_medians_hsv, instance_medians_rgb, instance_medians_lab, instance_to_npid, npid_to_instance



#def assign_nearby_instances(instance_to_item, buffer_range):
#    # Using morphological dilation to find nearby instances within a certain buffer range
#    kernel = np.ones((buffer_range, buffer_range), np.uint8)
#    return cv2.dilate(instance_to_item.astype(np.uint8), kernel).astype(bool)


def find_nearby_instance(instance_map, instance_to_npid, buffer_range=100):
    unique_instances = np.unique(instance_map)
    kernel = np.ones((buffer_range, buffer_range), np.uint8)

    nearby_instance = {}

    for instance_id in unique_instances:
        instance_mask = (instance_map == instance_id)

        if np.any(instance_mask):
            try:
                buffered_mask = (cv2.dilate(instance_mask.astype(np.uint8), kernel) - instance_mask).astype(bool)

                # Use boolean array to filter the integer array
                filtered_values = instance_map[buffered_mask]
                
                # Return unique values from the filtered array
                unique_values = np.unique(filtered_values)
                #nearby_instance[instance_to_npid[instance_id]] = unique_values

                if unique_values.shape[0] > 0:
                    # Create a vectorized function that maps dictionary keys to their values
                    vectorized_func = np.vectorize(lambda x: instance_to_npid[x])

                    # Apply the function to the array
                    nearby_instance[instance_to_npid[instance_id]] = vectorized_func(unique_values)
                else:
                    nearby_instance[instance_to_npid[instance_id]] = np.array([])

            except AttributeError:
                # (cv2.dilate(instance_mask.astype(np.uint8), kernel) - instance_mask) returns none...
                nearby_instance[instance_to_npid[instance_id]] = np.array([])
        else:
            nearby_instance[instance_to_npid[instance_id]] = np.array([])
    
    # Use the following script if we want to transform all at one time...
    # Create a vectorized function to map dictionary keys to their values
    #vectorized_func = np.vectorize(lambda x: instance_to_npid.get(x, x))

    # Use a dictionary comprehension to transform all arrays
    #transformed_nearby_instance = {key: vectorized_func(array) for key, array in nearby_instance.items()}

    #return transformed_nearby_instance
    return nearby_instance




def assign_instances_to_items(path_to_source_image, instance_map, target_hsv, target_rgb, target_lab, this_item_name, path_to_cropped_intermediate, 
                              max_threshold_hsv=30, max_threshold_rgb=30, max_threshold_lab=30, relaxation_factor=1.5, 
                              max_threshold_hsv_r2=45, max_threshold_rgb_r2=45, max_threshold_lab_r2=45, relaxation_factor_r2=2.0, 
                              buffer_range=10):
    # hyperparameter...
    # max_threshold_hsv=45, max_threshold_rgb=30, max_threshold_lab=30, relaxation_factor=1.33
    # max_threshold_hsv=60, max_threshold_rgb=45, max_threshold_lab=45, relaxation_factor=2.0


    # max_threshold_hsv=30, max_threshold_rgb=45, max_threshold_lab=30, relaxation_factor=1.5
    # max_threshold_hsv=60, max_threshold_rgb=30, max_threshold_lab=15, relaxation_factor=1.33
    # max_threshold_hsv=15, max_threshold_rgb=10, max_threshold_lab=5, relaxation_factor=1.1
    # max_threshold_hsv=10, max_threshold_rgb=10, max_threshold_lab=5  # max_threshold=30, relaxation_factor=1.1
    target_hsv = np.transpose(target_hsv, (1, 0))
    target_rgb = np.transpose(target_rgb, (1, 0))
    target_lab = np.transpose(target_lab, (1, 0))

    m = target_hsv.shape[1]
    unique_instances = np.unique(instance_map)
    n = len(unique_instances)
    #print('unique_instances', len(unique_instances), unique_instances)
    instance_to_item = np.zeros((n, m), dtype=bool)
    instance_to_item_r2 = np.zeros((n, m), dtype=bool)
    #relaxed_item_to_instance = np.zeros((m, n), dtype=bool)

    # A numpy array to record the count of items each instance is assigned to
    instance_item_counts = np.zeros(n, dtype=int)

    '''
    instance_medians = {} #[]
    instance_to_npid = {}
    npid_to_instance = {}
    
    output_hsv_image = np.zeros_like(hsv_image)  # Create an output image with the same size and type
    npid_counting = 0
    for instance_id in unique_instances:
        if instance_id == 0:
            continue
        mask = (instance_map == instance_id)
        # We transpose the image array to easily apply the mask
        hsv_masked = hsv_image[:, mask]
        if hsv_masked.size > 0:
            median_hsv = np.median(hsv_masked, axis=1)
        else:
            median_hsv = np.array([0, 0, 0]) # Placeholder for empty instances
        instance_medians[instance_id] = median_hsv
        
        # Set the pixels in the output image corresponding to this instance to the median HSV values
        for i in range(3):  # There are 3 channels: H, S, V
            output_hsv_image[i,:,:][mask] = median_hsv[i]
        
        #print('npid <-> instance_id: ', npid_counting,  instance_id)
        instance_to_npid[instance_id] = npid_counting
        npid_to_instance[npid_counting] = instance_id
        npid_counting += 1


    #instance_medians = np.array(instance_medians)
    output_hsv_image = np.transpose(output_hsv_image, (1, 2, 0))
    output_hsv_image = cv2.cvtColor(output_hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join('Temp', 'hsv_predict.png'), output_hsv_image)
    '''
    # replace the above with the following function
    instance_medians_hsv, instance_medians_rgb, instance_medians_lab, instance_to_npid, npid_to_instance = calculate_median_hsv(instance_map, path_to_source_image, this_item_name, path_to_cropped_intermediate)



    #nearby_instance_dict = find_nearby_instance(instance_map, instance_to_npid, buffer_range=100)



    #fill_instances_with_median_hsv(hsv_image, instance_map)
    
    #for i, median_hsv in enumerate(instance_medians):
    #    distances = np.sqrt(((target_hsv.T - median_hsv[:, np.newaxis]) ** 2).sum(axis=0))
    #    closest_item_ids = np.argwhere(distances < threshold).flatten()
    #    instance_to_item[i, closest_item_ids] = True

    #print('target_hsv', target_hsv)

    
    distances_to_nearest_item = np.full((n, m), np.inf)  # Initialize with infinities
    for i_of_instance, median_hsv in instance_medians_hsv.items():
        median_rgb = instance_medians_rgb[i_of_instance]
        median_lab = instance_medians_lab[i_of_instance]

        i_of_npid = instance_to_npid[i_of_instance]

        # HSV color space
        # Median HSV is (3,), need to reshape to (1, 3) to properly broadcast with target_hsv which is (3, m)
        hue_distance = np.abs(target_hsv[0, :] - median_hsv[0])
        #hsv_distance = np.sqrt(np.sum((target_hsv - median_hsv.reshape(3, 1)) ** 2, axis=0))
        #distances_hsv = (hue_distance + hsv_distance) / 2
        distances_hsv = hue_distance

        min_distance_hsv = np.min(distances_hsv)
        adaptive_threshold_hsv = min(min_distance_hsv * relaxation_factor, max_threshold_hsv)
        close_item_ids_hsv = distances_hsv <= adaptive_threshold_hsv

        ### relaxation
        adaptive_threshold_hsv_r2 = min(min_distance_hsv * relaxation_factor_r2, max_threshold_hsv_r2)
        close_item_ids_hsv_r2 = distances_hsv <= adaptive_threshold_hsv_r2


        # RGB color space
        distance_rgb = np.sqrt(np.sum((target_rgb - median_rgb.reshape(3, 1)) ** 2, axis=0))

        min_distance_rgb = np.min(distance_rgb)
        adaptive_threshold_rgb = min(min_distance_rgb * relaxation_factor, max_threshold_rgb)
        close_item_ids_rgb = distance_rgb <= adaptive_threshold_rgb

        ### relaxation
        adaptive_threshold_rgb_r2 = min(min_distance_rgb * relaxation_factor_r2, max_threshold_rgb_r2)
        close_item_ids_rgb_r2 = distance_rgb <= adaptive_threshold_rgb_r2


        # LAB color space
        distance_lab = np.abs(target_lab[1, :] - median_lab[1]) + np.abs(target_lab[2, :] - median_lab[2])

        min_distance_lab = np.min(distance_lab)
        adaptive_threshold_lab = min(min_distance_lab * relaxation_factor, max_threshold_lab)
        close_item_ids_lab = distance_lab <= adaptive_threshold_lab

        ### relaxation
        adaptive_threshold_lab_r2 = min(min_distance_lab * relaxation_factor_r2, max_threshold_lab_r2)
        close_item_ids_lab_r2 = distance_lab <= adaptive_threshold_lab_r2


        distances_to_nearest_item[i_of_npid] = distances_hsv


        #close_item_ids = (close_item_ids_hsv | close_item_ids_rgb | close_item_ids_lab)
        #instance_to_item[i_of_npid] = close_item_ids
        #instance_to_item[i_of_npid] = close_item_ids_rgb
        #instance_to_item[i_of_npid] = close_item_ids_lab
        instance_to_item[i_of_npid] = (close_item_ids_rgb | (close_item_ids_lab & close_item_ids_hsv))
        instance_to_item_r2[i_of_npid] = (close_item_ids_rgb_r2 | (close_item_ids_lab_r2 & close_item_ids_hsv_r2))


        '''
        # Initialize the boolean array with False values
        this_nearby_instance_dict = np.zeros(n, dtype=bool)

        # Directly set the relevant indices to True based on nearby_instance_dict[i_of_instance]
        try:
            this_nearby_instance_dict[nearby_instance_dict[i_of_npid]] = True
        except IndexError:
            # there no highlighted nearby instances...
            pass

        # Assign this array to the relaxed_instance_to_item for the current i_of_npid
        relaxed_item_to_instance[instance_to_item[i_of_npid]] |= this_nearby_instance_dict
        '''

        # Update counts for each instance
        instance_item_counts[i_of_npid] = np.sum(instance_to_item[i_of_npid])

        print('instance: ', i_of_npid, median_hsv, median_rgb, median_lab)
    
    # Applying the buffer range to include nearby instances
    #instance_to_item = assign_nearby_instances(instance_to_item, buffer_range)
        

    #return instance_to_item, relaxed_item_to_instance, distances_to_nearest_item, instance_to_npid, npid_to_instance, instance_item_counts
    #return instance_to_item, distances_to_nearest_item, instance_to_npid, npid_to_instance, instance_item_counts
    return instance_to_item, instance_to_item_r2, distances_to_nearest_item, instance_to_npid, npid_to_instance, instance_item_counts



#def create_binary_masks_for_items(instance_map, instance_to_item, instance_to_npid, npid_to_instance, instance_check, image_crop_size=1024):
def create_binary_masks_for_items(instance_map, instance_to_item, instance_to_item_r2, instance_to_npid, npid_to_instance, instance_check, image_crop_size=1024):
    # Number of items
    m = instance_to_item.shape[1]

    if instance_check == False:
        binary_masks = np.zeros((m, image_crop_size, image_crop_size), dtype=np.uint8)
        #return binary_masks
        return binary_masks, binary_masks

    # Size of the image
    size = instance_map.shape
    # Prepare container for binary masks
    binary_masks = np.zeros((m, size[0], size[1]), dtype=np.uint8)
    binary_masks_r2 = np.zeros((m, size[0], size[1]), dtype=np.uint8)
    #relaxed_binary_masks = np.zeros((m, size[0], size[1]), dtype=np.uint8)

    '''
    # Iterate through each item
    for item_index in range(m):
        # Find instances associated with this item
        associated_instances = np.where(instance_to_item[:, item_index])[0]
        # Create binary mask for this item
        for instance_id in associated_instances:
            binary_masks[item_index] |= (instance_map == instance_id)
    '''
    

    # Unique instance IDs for the whole map
    unique_instances = np.unique(instance_map)
    unique_instances = unique_instances[unique_instances!=0]

    '''
    # Check whether the instances cover all the image
    acc_mask = np.zeros((instance_map.shape[0], instance_map.shape[1]), dtype=np.uint8)
    for instance_id in unique_instances:
        #mask = (instance_map == instance_id)
        this_mask = np.zeros((instance_map.shape[0], instance_map.shape[1]), dtype=np.uint8)
        this_mask[instance_map==instance_id] = 255
        acc_mask = cv2.bitwise_or(acc_mask, this_mask)
        cv2.imwrite(os.path.join('Temp', str(instance_id)+'_get.png'), this_mask)
    cv2.imwrite(os.path.join('Temp', 'Acc_get.png'), acc_mask)
    '''


    
    # Create a mask for each instance only once
    #instance_masks = {instance_id: (instance_map == instance_id) for instance_id in unique_instances}
    instance_masks = {instance_to_npid[instance_id]: (instance_map == instance_id) for instance_id in unique_instances}

    # Iterate through each item
    for item_index in range(m):
        # Find instances associated with this item
        associated_instances = np.where(instance_to_item[:, item_index])[0]
        # Use a logical OR to combine masks of the associated instances
        item_mask = np.any([instance_masks[inst] for inst in associated_instances if inst in instance_masks], axis=0)
        binary_masks[item_index] = item_mask.astype(np.uint8) * 255

        # Find instances associated with this item
        associated_instances_r2 = np.where(instance_to_item_r2[:, item_index])[0]
        # Use a logical OR to combine masks of the associated instances
        item_mask_r2 = np.any([instance_masks[inst] for inst in associated_instances_r2 if inst in instance_masks], axis=0)
        binary_masks_r2[item_index] = item_mask_r2.astype(np.uint8) * 255

        '''
        # Find instances associated with this item
        relaxed_associated_instances = np.where(relaxed_item_to_index[item_index, :])[0]
        # Use a logical OR to combine masks of the associated instances
        relaxed_item_mask = np.any([instance_masks[inst] for inst in relaxed_associated_instances if inst in instance_masks], axis=0)
        relaxed_binary_masks[item_index] = relaxed_item_mask.astype(np.uint8) * 255
        '''

    return binary_masks, binary_masks_r2


def create_integer_masks(path_to_instance_map, instance_to_npid, instance_item_counts, image_crop_size=1024):
    # Size of the image
    instance_check = os.path.isfile(path_to_instance_map)
    try:
        instance_map = cv2.imread(path_to_instance_map, cv2.IMREAD_UNCHANGED)
        size = instance_map.shape

        # Create a mapping array where each instance ID maps to its count
        max_instance_id = np.max(instance_map)
        if max_instance_id is None:
            max_instance_id = size
        count_map = np.zeros(max_instance_id + 1, dtype=int)  # +1 because instance IDs are not zero-indexed

        # Get all unique instances from the instance map, excluding zero
        unique_instances = np.unique(instance_map)
        unique_instances = unique_instances[unique_instances != 0]

        # Map instance IDs to npids and then to counts
        npids = [instance_to_npid[id] for id in unique_instances]
        count_map[unique_instances] = instance_item_counts[npids]

        # Use the mapping array to create the integer_masks array
        integer_masks = count_map[instance_map]
    except:
        print('somehow error: ', path_to_instance_map)
        integer_masks = np.zeros((image_crop_size, image_crop_size), dtype=np.uint8)

    return integer_masks




def assigning_region_map(path_to_source_image, path_to_instance_map, color_list_indexed_hsv, color_list_indexed_rgb, color_list_indexed_lab, item_list_indexed, path_to_cropped_intermediate, path_to_cropped_predict, image_crop_size=1024):
    this_split_id = '_'.join(os.path.basename(path_to_source_image).replace('_sup_0.png', '').split('_')[-2:])

    #print(path_to_instance_map)
    instance_check = os.path.isfile(path_to_instance_map)
    instance_map = cv2.imread(path_to_instance_map, cv2.IMREAD_UNCHANGED)
    #print('CV2.IMREAD_UNCHANGED')
    #print('instance_map', instance_map.shape)
    #print(np.unique(instance_map))
    
    #print('color_list_indexed', color_list_indexed.shape)
    #bool_result, relaxed_bool_result, dist_result, instance_to_npid, npid_to_instance, instance_item_counts = assign_instances_to_items(path_to_source_image, instance_map, color_list_indexed_hsv, color_list_indexed_rgb, color_list_indexed_lab, os.path.basename(path_to_source_image).replace('_sup_0.png', ''), path_to_cropped_intermediate)
    #bool_result, dist_result, instance_to_npid, npid_to_instance, instance_item_counts = assign_instances_to_items(path_to_source_image, instance_map, color_list_indexed_hsv, color_list_indexed_rgb, color_list_indexed_lab, os.path.basename(path_to_source_image).replace('_sup_0.png', ''), path_to_cropped_intermediate)
    bool_result, bool_result_r2, dist_result, instance_to_npid, npid_to_instance, instance_item_counts = assign_instances_to_items(path_to_source_image, instance_map, color_list_indexed_hsv, color_list_indexed_rgb, color_list_indexed_lab, os.path.basename(path_to_source_image).replace('_sup_0.png', ''), path_to_cropped_intermediate)
    #os.path.join(path_to_target_map_5, 'color_info.csv')

    #print('bool_result', bool_result.shape)
    #binary_masks, relaxed_binary_masks = create_binary_masks_for_items(instance_map, bool_result, relaxed_bool_result, instance_to_npid, npid_to_instance, instance_check, image_crop_size)
    #binary_masks = create_binary_masks_for_items(instance_map, bool_result, instance_to_npid, npid_to_instance, instance_check, image_crop_size)
    binary_masks, binary_masks_r2 = create_binary_masks_for_items(instance_map, bool_result, bool_result_r2, instance_to_npid, npid_to_instance, instance_check, image_crop_size)
    #print(binary_masks.shape)

    for this_item in range(0, binary_masks.shape[0]):
        #cv2.imwrite(os.path.join(path_to_cropped_predict, item_list_indexed[this_item]+'_predict.png'), binary_masks[this_item])
        cv2.imwrite(os.path.join(path_to_cropped_predict, item_list_indexed[this_item]+'_'+this_split_id+'_predict.png'), binary_masks[this_item])
        
        # when printing the output with relaxed instances, exclude the original predicted instance
        cv2.imwrite(os.path.join(path_to_cropped_predict, item_list_indexed[this_item]+'_'+this_split_id+'_predict_r2_instance.png'), cv2.bitwise_and(binary_masks_r2[this_item], 255-binary_masks[this_item]))

        # when printing the output with relaxed instances, exclude the original predicted instance
        #cv2.imwrite(os.path.join(path_to_cropped_predict, item_list_indexed[this_item]+'_'+this_split_id+'_predict_relaxed_instance.png'), cv2.bitwise_and(relaxed_binary_masks[this_item], 255-binary_masks[this_item]))
    
    integer_masks = create_integer_masks(path_to_instance_map, instance_to_npid, instance_item_counts, image_crop_size)

    return True, binary_masks.shape[0], this_split_id, this_split_id.split('_')[0], this_split_id.split('_')[1], integer_masks







'''
def fill_instances_with_median_hsv(hsv_image, instance_map):
    #print(hsv_image.shape)
    #hsv_image = np.transpose(hsv_image, (2, 0, 1))
    print('hsv_image', hsv_image.shape)
    
    unique_instances = np.unique(instance_map)
    output_hsv_image = np.zeros_like(hsv_image)  # Create an output image with the same size and type

    for instance_id in unique_instances:
        if instance_id == 0:
            continue
        mask = (instance_map == instance_id)
        hsv_masked = hsv_image[:, mask]
        
        if hsv_masked.size > 0:
            median_hsv = np.median(hsv_masked, axis=1)  # Compute median HSV values for the pixels in this instance
        else:
            median_hsv = np.array([0, 0, 0])  # Placeholder for instances with no pixels (unlikely unless the map is incorrect)

        # Set the pixels in the output image corresponding to this instance to the median HSV values
        for i in range(3):  # There are 3 channels: H, S, V
            output_hsv_image[i,:,:][mask] = median_hsv[i]

    output_hsv_image = np.transpose(output_hsv_image, (1, 2, 0))
    output_hsv_image = cv2.cvtColor(output_hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join('Temp', 'hsv_predict.png'), output_hsv_image)
    
    return output_hsv_image
'''