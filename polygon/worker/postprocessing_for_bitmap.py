import numpy as np
import cv2
import os
import math

def postprocessing_for_bitmap(this_name, path_to_input, dir_to_output, suffix, color_channel, particular_channel, binarization, enhancement=False, resize_times=0, crop_size=1024):
    img = cv2.imread(path_to_input)
    if color_channel == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if resize_times > 0:
            crop_size = int(crop_size/(math.pow(2,resize_times)))
            img = cv2.resize(img, (int(img.shape[1]/math.pow(2,resize_times)), int(img.shape[0]/math.pow(2,resize_times))), interpolation=cv2.INTER_NEAREST)
        if enhancement == True:
            kernel = np.ones((3,3), np.uint8)
            img = 255-img
            img = cv2.dilate(img, kernel, iterations=1)
            #img = cv2.erode(img, kernel, iterations=1)
            img = 255-img
        empty_grid = np.zeros((crop_size, crop_size), dtype='uint8').astype(float)
    else:
        if particular_channel is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img[:, :, particular_channel]
            empty_grid = np.zeros((crop_size, crop_size), dtype='uint8').astype(float)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            empty_grid = np.zeros((crop_size, crop_size, 3), dtype='uint8')
    print(path_to_input, img.shape)
    


    for r in range(0,math.ceil(img.shape[0]/crop_size)):
        for c in range(0,math.ceil(img.shape[1]/crop_size)):
            this_output_file_name = os.path.join(dir_to_output, this_name+'_'+str(r)+'_'+str(c)+suffix+'.png')
            
            if (min(r*crop_size+crop_size, img.shape[0]) - r*crop_size <= 0) or (min(c*crop_size+crop_size, img.shape[1]) - c*crop_size <= 0):
                continue
            
            r_0 = r*crop_size
            r_1 = min(r*crop_size+crop_size, img.shape[0])
            c_0 = c*crop_size
            c_1 = min(c*crop_size+crop_size, img.shape[1])

            #print(r, c, r_0, r_1, c_0, c_1)
            if color_channel == 1:
                if r_1-r_0 < crop_size or c_1-c_0 < crop_size:
                    if r_1-r_0 < crop_size:
                        img_concat_temp = np.concatenate([img[r_0:r_1, c_0:c_1], empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0)]], axis=0)
                    else:
                        img_concat_temp = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
                    if c_1-c_0 < crop_size:
                        img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0)], empty_grid[:, 0:crop_size-(c_1-c_0)]], axis=1)
                    else:
                        img_concat = np.copy(img_concat_temp).astype(float)
                else:
                    img_concat = np.copy(img[r_0:r_1, c_0:c_1]).astype(float)
            else:
                if r_1-r_0 < crop_size or c_1-c_0 < crop_size:
                    if r_1-r_0 < crop_size:
                        img_concat_temp = np.concatenate([img[r_0:r_1, c_0:c_1, :], empty_grid[0:crop_size-(r_1-r_0), 0:(c_1-c_0), :]], axis=0)
                    else:
                        img_concat_temp = np.copy(img[r_0:r_1, c_0:c_1, :])
                    if c_1-c_0 < crop_size:
                        img_concat = np.concatenate([img_concat_temp[:, 0:(c_1-c_0), :], empty_grid[:, 0:crop_size-(c_1-c_0), :]], axis=1)
                    else:
                        img_concat = np.copy(img_concat_temp)
                else:
                    img_concat = np.copy(img[r_0:r_1, c_0:c_1, :])
            

            if binarization:
                img_concat[img_concat > 0] = 255
            if color_channel == 3:
                img_concat = cv2.cvtColor(img_concat, cv2.COLOR_RGB2BGR)
            cv2.imwrite(this_output_file_name, img_concat)
    
    return math.ceil(img.shape[0]/crop_size), math.ceil(img.shape[1]/crop_size)
