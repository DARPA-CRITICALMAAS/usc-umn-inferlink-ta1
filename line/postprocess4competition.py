import os
import sys
import cv2
import numpy as np

def end_pt(img, x, y):
    # check 8-dir, if only one connection, it's an end pt
    subreg = img[max(0, x-1):min(x+2, img.shape[0]), \
                 max(0, y-1):min(y+2, img.shape[1])]
    if np.sum(subreg) <= 2: 
        return True
    return False

def conn4dir(img, x, y):
    # check the connections in four directions
    count = 0
    if img[x-1, y] == 1:
        count += 1
    if img[x+1, y] == 1:
        count += 1
    if img[x, y+1] == 1:
        count += 1
    if img[x, y-1] == 1:
        count += 1
    # if < 2, needs buffering  
    if count < 2 and not end_pt(img, x, y):
        return False
    # else, no needs buffering
    return True

def buff_pt(img, x, y):
    # check four diagonal directions
    # buffer in the 4 connections
    # return (x, y), which needs for buffering
    if img[x-1, y-1] == 1:
        if img[x-1, y] == 0:
            return (x-1, y)
        if img[x, y-1] == 0:
            return (x, y-1)
    if img[x-1, y+1] == 1:
        if img[x-1, y] == 0:
            return (x-1, y)
        if img[x, y+1] == 0:
            return (x, y+1)
    if img[x+1, y-1] == 1:
        if img[x, y-1] == 0:
            return (x, y-1)
        if img[x+1, y] == 0:
            return (x+1, y)
    if img[x+1, y+1] == 1:
        if img[x, y+1] == 0:
            return (x, y+1)
        if img[x+1, y] == 0:
            return (x+1, y)
    return (x, y)

def buff(img):
    buf_img = np.zeros_like(img)
    nz_indices = np.where(img!=0)
    for i in range(nz_indices[0].size):
        x, y = nz_indices[0][i], nz_indices[1][i]
        if not conn4dir(img, x, y):
            bf_x, bf_y = buff_pt(img, x, y)
            buf_img[bf_x, bf_y] = 1
        buf_img[x, y] = 1
    return buf_img

if __name__ == '__main__':
    map_image_dir = '/data/weiweidu/criticalmaas_data/validation_fault_line_comb'
    pred_image_dir = '/data/weiweidu/LDTR_criticalmaas_gradient/pred_maps'
    
    selected_maps = []

    for root, dirs, files in os.walk(pred_image_dir, topdown=False):
        for map_name in files:
            if '_buf' in map_name:
                continue
            if selected_maps != [] and map_name[:-4] not in selected_maps:
                continue
            print('----- buffering {} -----'.format(map_name[:-4]))
            pred_img = cv2.imread(os.path.join(root, map_name), 0) / 255
            buf_pred_img = buff(pred_img)
            cv2.imwrite(os.path.join(root, map_name[:-4]+'_buf.png'), buf_pred_img*255)