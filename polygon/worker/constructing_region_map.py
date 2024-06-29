### The source code is adpated from https://github.com/hepesu/LineFiller
import cv2
import os
import numpy as np
from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, show_fill_map
from linefiller.thinning import thinning

def constructing_region_map(path_to_input, path_to_output, is_region=True, image_crop_size=1024):
    im = cv2.imread(path_to_input, cv2.IMREAD_GRAYSCALE)
    #print(im.shape)
    ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)
    if np.unique(binary).shape[0] == 1:
        #place_holder = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
        place_holder = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
        cv2.imwrite(path_to_output, place_holder)
        return True, path_to_input

    fills = []
    result = binary

    fill = trapped_ball_fill_multi(result, 3, method='max')
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 2, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = trapped_ball_fill_multi(result, 1, method=None)
    fills += fill
    result = mark_fill(result, fill)

    fill = flood_fill_multi(result)
    fills += fill

    fillmap = build_fill_map(result, fills)
    print(fillmap.shape, np.unique(fillmap).shape[0])

    fillmap = merge_fill(fillmap)

    #cv2.imwrite(path_to_output, show_fill_map(thinning(fillmap)))
    
    #fillmap = thinning(fillmap)
    #if fillmap.shape[0] != 1024 or fillmap.shape[1] != 1024:
    #    fillmap = cv2.resize(fillmap, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    #cv2.imwrite(path_to_output, fillmap.astype(np.uint16))

    if is_region == True:
        fillmap = thinning(fillmap)
        cv2.imwrite(path_to_output.replace('.png', '_temp_fill.tif'), fillmap.astype(np.uint16))
        cv2.imwrite(path_to_output.replace('.png', '_temp_color.png'), show_fill_map(fillmap))
        #cv2.imwrite(path_to_output, fillmap.astype(np.uint16))

        # Read the image back in 16-bit
        img_read = cv2.imread(path_to_output.replace('.png', '_temp_fill.tif'), cv2.IMREAD_UNCHANGED)
        print('CV2.IMREAD_UNCHANGED')
        print(img_read.shape)
        print(np.unique(img_read))

        return True, path_to_output.replace('.png', '_temp_fill.tif')
    else:
        cv2.imwrite(path_to_output.replace('.png', '_temp_fill.tif'), fillmap.astype(np.uint16))

        fillmap = thinning(fillmap)
        fillmap = show_fill_map(fillmap).astype(np.uint8)
        if fillmap.shape[0] != image_crop_size or fillmap.shape[1] != image_crop_size:
            fillmap = cv2.resize(fillmap, (image_crop_size, image_crop_size), interpolation=cv2.INTER_NEAREST)
            kernel_size = 3
            fillmap = cv2.GaussianBlur(fillmap, (kernel_size, kernel_size), 0)
        linemap = cv2.Canny(fillmap, 1, 10)
        #linemap = np.where(fillmap > 0, 0, 255).astype(np.uint8)

        # we only keep the boundary from the fillmap
        #fillmap = thinning(fillmap)
        #linemap = np.where(fillmap > 0, 0, 255)
        cv2.imwrite(path_to_output, linemap)
    return True, path_to_input