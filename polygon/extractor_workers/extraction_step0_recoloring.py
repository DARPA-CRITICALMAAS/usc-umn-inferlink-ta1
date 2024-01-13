
import numpy as np
import scipy.ndimage
import tracemalloc


def extraction_step0_recoloring(this_grid, sub_rgb_rb, color_avg, color_avg2, mapping_color_set_to_color_prob_tp):
    tracemalloc.start()
    ### only process a small part of the whole subregion
    #im = np.copy(rgb_rb[ystart:ystop, xstart:xstop, :])
    im = np.copy(sub_rgb_rb)
    image = im.reshape(im.shape[0],im.shape[1],1,3)

    distance_kernel = np.ones((5,5)) / 25.0
    distance_kernel = distance_kernel[:, :, None]

    # Create color container 
    colors_container = np.ones(shape=[image.shape[0],image.shape[1],len(color_avg),3])
    for i,color in enumerate(color_avg):
        colors_container[:,:,i,:] = color
    colors_container2 = np.ones(shape=[image.shape[0],image.shape[1],len(color_avg2),3])
    for i,color in enumerate(color_avg2):
        colors_container2[:,:,i,:] = color
    
    rgb_weight = np.ones(shape=[image.shape[0],image.shape[1],1,3])
    rgb_weight[:,:,:,0] = 1 # 2
    rgb_weight[:,:,:,1] = 1 # 4
    rgb_weight[:,:,:,2] = 1 # 3

    background_correction_direct_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
    background_correction_direct_rgb[:,:,:,0] = 1.0
    background_correction_direct_rgb[:,:,:,1] = 1.0
    background_correction_direct_rgb[:,:,:,2] = 1.0

    image_deviation = np.zeros(shape=[image.shape[0],image.shape[1],1,3])
    image_deviation[:,:,:,0] = image[:,:,:,0] - image[:,:,:,1]
    image_deviation[:,:,:,1] = image[:,:,:,0] - image[:,:,:,2]
    image_deviation[:,:,:,2] = image[:,:,:,1] - image[:,:,:,2]

    legend_deviation = np.zeros(shape=[image.shape[0],image.shape[1],len(color_avg),3])
    legend_deviation[:,:,:,0] = colors_container[:,:,:,0] - colors_container[:,:,:,1]
    legend_deviation[:,:,:,1] = colors_container[:,:,:,0] - colors_container[:,:,:,2]
    legend_deviation[:,:,:,2] = colors_container[:,:,:,1] - colors_container[:,:,:,2]
    
    background_correction_deviated_rgb = np.ones(shape=[image.shape[0],image.shape[1],1,3])
    background_correction_deviated_rgb[:,:,:,:] = 0.5 + 0.5*(1.0-abs(image_deviation[:,:,:,:])/255.0)


    def closest(image,color_container):
        shape = image.shape[:2]
        total_shape = shape[0]*shape[1]

        # calculate distances
        distances_0 = np.sqrt(np.sum(rgb_weight*((color_container*background_correction_direct_rgb-image)**2),axis=3))
        distances_1 = np.sqrt(np.sum(((legend_deviation*background_correction_deviated_rgb-image_deviation)**2),axis=3))
        distances = distances_0*0.95 + distances_1*0.05

        # in the 1st version, the distance is the distance to the color of each key
        # in the 2nd version, the distance is the distance to the color under the color set of each key

        #print(distances.shape) # shape: (1500, 1500, # of colors)
        #print(mapping_color_set_to_color_prob_tp.shape) # shape: (# of colors, # of keys)

        multiplied_distances = np.dot(distances, mapping_color_set_to_color_prob_tp)

        #print(multiplied_distances.shape) # shape: (1500, 1500, # of keys)
        
        conv_distances = scipy.ndimage.convolve(multiplied_distances, distance_kernel)


        min_index_map = np.argmin(conv_distances, axis=2)
        min_index = min_index_map.reshape(-1)
        natural_index = np.arange(total_shape)

        reshaped_container = colors_container2.reshape(-1,len(color_avg2),3) # only use one color to re-color the map

        color_view = reshaped_container[natural_index, min_index].reshape(shape[0], shape[1], 3)
        return color_view, min_index_map
    
    
    result_image, min_index_map = closest(image, colors_container)
    result_image = result_image.astype(np.uint8)
    min_index_map = min_index_map.astype(np.uint8)

    peak_ram_usage = tracemalloc.get_traced_memory()[1]/1000000000
    tracemalloc.stop()

    return this_grid, result_image, min_index_map, peak_ram_usage
