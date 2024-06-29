import cv2
import os
import numpy as np

def reading_base_for_integration(path_to_merged_predict, this_info):
    this_empty_grid = cv2.imread(os.path.join(path_to_merged_predict, this_info[0]+'_predict.png'))
    this_empty_grid = cv2.cvtColor(this_empty_grid, cv2.COLOR_BGR2GRAY)
    this_empty_grid[this_empty_grid>0] = 1
    #print(this_info[1], np.unique(this_empty_grid))
    #empty_counter[this_info[1]] = this_empty_grid

    return True, this_info[1], this_empty_grid