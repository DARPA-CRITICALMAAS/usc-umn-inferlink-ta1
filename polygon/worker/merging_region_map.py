import cv2
import numpy as np
import worker.constructing_region_map as constructing_region_map
from scipy.ndimage import binary_dilation, generate_binary_structure
from linefiller.trappedball_fill import show_fill_map

def merge_small_clusters(image, min_size=500, merge_factor=5):
    # Label the clusters and find sizes
    unique, counts = np.unique(image, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    # Identify small clusters
    small_clusters = {cluster_id: size for cluster_id, size in cluster_sizes.items() if size < min_size}

    # Preparing the structure for dilation (8-connectivity)
    structure = generate_binary_structure(2, 2)

    # Process each small cluster
    for small_cluster, small_size in small_clusters.items():
        # Create a mask for the small cluster
        mask = (image == small_cluster)
        
        # Dilate the mask to get the surrounding area
        dilated_mask = binary_dilation(mask, structure=structure)
        surrounding_mask = dilated_mask & ~mask

        # Check if the cluster is on the boundary
        if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
            continue  # Skip this cluster if it touches the boundary

        # Get the IDs of the surrounding clusters
        surrounding_ids = np.unique(image[surrounding_mask])
        
        if surrounding_ids.size == 1:
            # Only one surrounding cluster, check its size
            surrounding_cluster = surrounding_ids[0]
            surrounding_size = cluster_sizes.get(surrounding_cluster, 0)
            
            if surrounding_size >= merge_factor * small_size:
                # The surrounding cluster is large enough, merge clusters
                image[mask] = surrounding_cluster
                # Update the size dictionary
                cluster_sizes[surrounding_cluster] += small_size
                cluster_sizes[small_cluster] = 0

    # Final step: Remap IDs to be continuous
    final_ids = np.unique(image)
    remap_dict = {old_id: new_id for new_id, old_id in enumerate(final_ids, start=1)}
    remapped_image = np.vectorize(remap_dict.get)(image)

    return remapped_image




def merging_region_map(path_to_input, path_to_output, scale_count, image_crop_size=1024):
    base_linemap = np.zeros((image_crop_size, image_crop_size), dtype=np.uint8)
    for this_scale in range(0, scale_count):
        # Read the image back in 16-bit
        #img_read = cv2.imread(path_to_input.replace('_sup_0.png', '_sup_'+str(this_scale)+'.png'), cv2.IMREAD_UNCHANGED)

        input_linemap = cv2.imread(path_to_input.replace('_sup_0.png', '_sup_'+str(this_scale)+'.png'), cv2.IMREAD_GRAYSCALE)

        # if the shape of the new linemap (of compressed scale) is not 1024x1024, there is not additional information (pixels not equal to 0)
        if input_linemap.shape[0] != image_crop_size or input_linemap.shape[1] != image_crop_size:
            continue

        # candidate region for including new boundaries is based on the existing boundaries
        # areas nearby existing boundaries are not considered for new boundaries
        kernel = np.ones((10, 10), np.uint8)
        candidate_area = cv2.dilate(base_linemap, kernel, iterations=1)
        candidate_input_linemap = cv2.bitwise_and(input_linemap, 255-candidate_area)
        
        # buffer the legit new boundaries to help connecting to existing boundaries
        kernel = np.ones((3, 3), np.uint8)
        candidate_input_linemap = cv2.dilate(candidate_input_linemap, kernel, iterations=1)
        input_linemap = cv2.bitwise_and(input_linemap, candidate_input_linemap)

        base_linemap = 255 - cv2.bitwise_or(base_linemap, input_linemap)
    
    cv2.imwrite(path_to_output, base_linemap)
    cv2.imwrite(path_to_output.replace('.png', '_line.png'), 255-base_linemap)
    _, path_to_updated_output = constructing_region_map.constructing_region_map(path_to_output, path_to_output, True, image_crop_size)


    # merge small instances
    img_read = cv2.imread(path_to_updated_output, cv2.IMREAD_UNCHANGED)
    merged_img = merge_small_clusters(img_read)
    cv2.imwrite(path_to_updated_output.replace('5_temp_fill.tif', '6_temp_fill.tif'), merged_img.astype(np.uint16))
    cv2.imwrite(path_to_updated_output.replace('5_temp_fill.tif', '6_temp_color.png'), show_fill_map(merged_img))

    return True, path_to_input