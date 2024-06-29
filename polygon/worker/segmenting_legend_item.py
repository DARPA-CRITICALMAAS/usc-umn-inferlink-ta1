import numpy as np
import cv2
import os

ITEM_WIDTH = 100
ITEM_HEIGHT = 50

def get_center_of_mass(this_item):
    # Find indices where the value is = 0
    indices = np.argwhere(this_item == 0)
    if len(indices) == 0:
        return [int(this_item.shape[0]/2.0), int(this_item.shape[1]/2.0)]
    
    # Calculate the center of mass
    center = np.mean(indices, axis=0, dtype=int)
    return tuple(center)


def segmenting_legend_item(dir_to_map, map_name, dir_to_item_output, item_name, this_item):
    cv2.imwrite(os.path.join(dir_to_item_output, map_name.replace('.tif','')+'_'+item_name+'.tif'), this_item)
    this_item = cv2.cvtColor(this_item, cv2.COLOR_BGR2GRAY)
    # text-pattern matching is not needed
    '''
    # find the dominated color in this legend item (if white => text as black; if black => text as white)
    dominated_color = np.median(this_item[int(this_item.shape[0]*0.1):int(this_item.shape[0]*0.9), int(this_item.shape[1]*0.1):int(this_item.shape[1]*0.9)])
    if dominated_color < 90:
        this_item = 255-this_item
    lower_bound = np.array([0])
    upper_bound = np.array([90])
    mask_box_legend = cv2.inRange(this_item, lower_bound, upper_bound)
    '''

    #print(this_item.shape)
    cropped_item = np.ones((ITEM_HEIGHT, ITEM_WIDTH), dtype=np.uint8)*255

    #print(this_item.shape, cropped_item.shape)
    #center_point = [int(this_item.shape[0]/2.0), int(this_item.shape[1]/2.0)]
    center_point = get_center_of_mass(this_item[int(this_item.shape[0]*0.1):int(this_item.shape[0]*0.9), int(this_item.shape[1]*0.1):int(this_item.shape[1]*0.9)])
    center_point = [int(center_point[0])+int(this_item.shape[0]*0.1), int(center_point[1])+int(this_item.shape[1]*0.1)]
    #print(center_point, this_item.shape)

    target_source = this_item[max(0, center_point[0]-int(ITEM_HEIGHT/2.0)):min(int(this_item.shape[0]/2.0)*2,center_point[0]+int(ITEM_HEIGHT/2.0)), max(0, center_point[1]-int(ITEM_WIDTH/2.0)):min(int(this_item.shape[1]/2.0)*2,center_point[1]+int(ITEM_WIDTH/2.0))]
    padding_x = 0
    padding_y = 0
    if target_source.shape[0] % 2 == 1:
        padding_x = 1
    if target_source.shape[1] % 2 == 1:
        padding_y = 1
    target_source = target_source[padding_x:, padding_y:]
    
    cropped_item[max(0, int(ITEM_HEIGHT/2.0)-int(target_source.shape[0]/2.0)):min(ITEM_HEIGHT, int(ITEM_HEIGHT/2.0)+int(target_source.shape[0]/2.0)), max(0, int(ITEM_WIDTH/2.0)-int(target_source.shape[1]/2.0)):min(ITEM_WIDTH, int(ITEM_WIDTH/2.0)+int(target_source.shape[1]/2.0))] = target_source[:, :]
    #cropped_item[max(0, int(ITEM_HEIGHT/2.0)-center_point[0]):min(ITEM_HEIGHT, int(ITEM_HEIGHT/2.0)+center_point[0]), max(0, int(ITEM_WIDTH/2.0)-center_point[1]):min(ITEM_WIDTH, int(ITEM_WIDTH/2.0)+center_point[1])] = \
    #    this_item[max(0, center_point[0]-int(ITEM_HEIGHT/2.0)):min(this_item.shape[0],center_point[0]+int(ITEM_HEIGHT/2.0)), max(0, center_point[1]-int(ITEM_WIDTH/2.0)):min(this_item.shape[1],center_point[1]+int(ITEM_WIDTH/2.0))]
    cv2.imwrite(os.path.join(dir_to_item_output, map_name.replace('.tif','')+'_'+item_name+'_cropped.tif'), cropped_item)

    #this_legend_item = cv2.imread(os.path.join(dir_to_item_output, map_name.replace('.tif','')+'_'+item_name+'_cropped.tif'))
    #this_legend_item = cv2.cvtColor(this_legend_item, cv2.COLOR_BGR2HSV)
    #cv2.imwrite(os.path.join(dir_to_item_output, map_name.replace('.tif','')+'_'+item_name+'_cropped_hsv.tif'), this_legend_item)

    return True, map_name, item_name