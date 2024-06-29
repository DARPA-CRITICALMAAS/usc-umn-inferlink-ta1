import math
import os
import cv2
import numpy as np

def merging_from_bitmap_v3(this_item_name, this_item_id, path_to_cropped_predict, path_to_merged_predict, img_shape, crop_size=1024):
    # this_item_name = map_name + legend_name

    empty_grid = np.zeros((img_shape[0], img_shape[1]), dtype='uint8').astype(float)
    empty_grid_relaxed = np.zeros((img_shape[0], img_shape[1]), dtype='uint8').astype(float)
    empty_flag = True

    for r in range(0,math.ceil(img_shape[0]/crop_size)):
        for c in range(0,math.ceil(img_shape[1]/crop_size)):
            this_block_source = os.path.join(path_to_cropped_predict, this_item_name+'_'+str(r)+'_'+str(c)+'_predict.png')
            this_block_source_relaxed = os.path.join(path_to_cropped_predict, this_item_name+'_'+str(r)+'_'+str(c)+'_predict_r2_instance.png')
            #print(this_block_source)
            already_predicted = os.path.isfile(this_block_source)
            already_predicted_relaxed = os.path.isfile(this_block_source_relaxed)

            if already_predicted == True and already_predicted_relaxed == True:
                block_img = cv2.imread(this_block_source)
                block_img = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)
                block_img_relaxed = cv2.imread(this_block_source_relaxed)
                block_img_relaxed = cv2.cvtColor(block_img_relaxed, cv2.COLOR_BGR2GRAY)

                r_0 = r*crop_size
                r_1 = min(r*crop_size+crop_size, img_shape[0])
                c_0 = c*crop_size
                c_1 = min(c*crop_size+crop_size, img_shape[1])
                
                empty_grid[r_0:r_1, c_0:c_1] = block_img[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
                empty_grid_relaxed[r_0:r_1, c_0:c_1] = block_img_relaxed[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
            else:
                pass
                #empty_flag = False
                #break
        if empty_flag == False:
            break
    
    if empty_flag == True:
        # dilate the final output...
        #kernel = np.ones((2,2), np.uint8)
        #empty_grid = cv2.erode(empty_grid, kernel, iterations=1)

        #kernel = np.ones((3,3), np.uint8)
        #empty_grid = cv2.dilate(empty_grid, kernel, iterations=1)
        empty_grid[empty_grid > 0] = 255
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict.png'), empty_grid)
        empty_grid_relaxed[empty_grid_relaxed > 0] = 255
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_r2_instance.png'), empty_grid_relaxed)

        empty_grid[empty_grid > 0] = 1
        empty_grid_relaxed[empty_grid_relaxed > 0] = 1
        return True, this_item_name, this_item_id#, empty_grid
    else:
        return False, this_item_name, this_item_id#, empty_grid
    

def merging_from_bitmap_v2(this_item_name, this_item_id, path_to_cropped_predict, path_to_merged_predict, img_shape, crop_size=1024):
    # this_item_name = map_name + legend_name

    empty_grid = np.zeros((img_shape[0], img_shape[1]), dtype='uint8').astype(float)
    empty_grid_relaxed = np.zeros((img_shape[0], img_shape[1]), dtype='uint8').astype(float)
    empty_flag = True

    for r in range(0,math.ceil(img_shape[0]/crop_size)):
        for c in range(0,math.ceil(img_shape[1]/crop_size)):
            this_block_source = os.path.join(path_to_cropped_predict, this_item_name+'_'+str(r)+'_'+str(c)+'_predict.png')
            this_block_source_relaxed = os.path.join(path_to_cropped_predict, this_item_name+'_'+str(r)+'_'+str(c)+'_predict_relaxed_instance.png')
            #print(this_block_source)
            already_predicted = os.path.isfile(this_block_source)
            already_predicted_relaxed = os.path.isfile(this_block_source_relaxed)

            if already_predicted == True and already_predicted_relaxed == True:
                block_img = cv2.imread(this_block_source)
                block_img = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)
                block_img_relaxed = cv2.imread(this_block_source_relaxed)
                block_img_relaxed = cv2.cvtColor(block_img_relaxed, cv2.COLOR_BGR2GRAY)

                r_0 = r*crop_size
                r_1 = min(r*crop_size+crop_size, img_shape[0])
                c_0 = c*crop_size
                c_1 = min(c*crop_size+crop_size, img_shape[1])
                
                empty_grid[r_0:r_1, c_0:c_1] = block_img[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
                empty_grid_relaxed[r_0:r_1, c_0:c_1] = block_img_relaxed[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
            else:
                pass
                #empty_flag = False
                #break
        if empty_flag == False:
            break
    
    if empty_flag == True:
        # dilate the final output...
        #kernel = np.ones((2,2), np.uint8)
        #empty_grid = cv2.erode(empty_grid, kernel, iterations=1)

        #kernel = np.ones((3,3), np.uint8)
        #empty_grid = cv2.dilate(empty_grid, kernel, iterations=1)
        empty_grid[empty_grid > 0] = 255
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict.png'), empty_grid)
        empty_grid_relaxed[empty_grid_relaxed > 0] = 255
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict_relaxed_instance.png'), empty_grid_relaxed)

        empty_grid[empty_grid > 0] = 1
        empty_grid_relaxed[empty_grid_relaxed > 0] = 1
        return True, this_item_name, this_item_id#, empty_grid
    else:
        return False, this_item_name, this_item_id#, empty_grid
    

def merging_from_bitmap(this_item_name, this_item_id, path_to_cropped_predict, path_to_merged_predict, img_shape, crop_size=1024):
    # this_item_name = map_name + legend_name

    empty_grid = np.zeros((img_shape[0], img_shape[1]), dtype='uint8').astype(float)
    empty_flag = True

    for r in range(0,math.ceil(img_shape[0]/crop_size)):
        for c in range(0,math.ceil(img_shape[1]/crop_size)):
            this_block_source = os.path.join(path_to_cropped_predict, this_item_name+'_'+str(r)+'_'+str(c)+'_predict.png')
            #print(this_block_source)
            already_predicted = os.path.isfile(this_block_source)

            if already_predicted == True:
                block_img = cv2.imread(this_block_source)
                block_img = cv2.cvtColor(block_img, cv2.COLOR_BGR2GRAY)

                r_0 = r*crop_size
                r_1 = min(r*crop_size+crop_size, img_shape[0])
                c_0 = c*crop_size
                c_1 = min(c*crop_size+crop_size, img_shape[1])
                
                empty_grid[r_0:r_1, c_0:c_1] = block_img[r_0%crop_size:(r_1-r_0), c_0%crop_size:(c_1-c_0)]
            else:
                pass
                #empty_flag = False
                #break
        if empty_flag == False:
            break
    
    if empty_flag == True:
        # dilate the final output...
        #kernel = np.ones((2,2), np.uint8)
        #empty_grid = cv2.erode(empty_grid, kernel, iterations=1)

        #kernel = np.ones((3,3), np.uint8)
        #empty_grid = cv2.dilate(empty_grid, kernel, iterations=1)
        empty_grid[empty_grid > 0] = 255
        cv2.imwrite(os.path.join(path_to_merged_predict, this_item_name+'_predict.png'), empty_grid)

        empty_grid[empty_grid > 0] = 1
        return True, this_item_name, this_item_id#, empty_grid
    else:
        return False, this_item_name, this_item_id#, empty_grid