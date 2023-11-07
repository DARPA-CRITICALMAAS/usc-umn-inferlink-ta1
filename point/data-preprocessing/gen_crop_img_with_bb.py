import cv2
import numpy as np
from PIL import ImageFile
from PIL import Image 
import os
import json
Image.MAX_IMAGE_PIXELS = 1000000000 

'''
This code is to generate dataset for object detection.
Input dir: TA1 feature extraction training directory
Output dir: cropped images patches with each BB and class (txt) 

Process:
Read each map sheet and point masks 
the way to get the point masks name : substract the name of PT with Map sheet 
Generate a dictionary of entire point symbols (save them with the txt files)
Set the cropping size 1024*1024, and if there are points contained in the mask, draw a BB with them and add idx of class
BB size = 60*60
Cropped output image/txt name format : MAPSHEETNAME_HN_WH.jpg and MAPSHEETNAME_HN_WH.txt
'''


input_root='./INPUT/DIR'
output_root='./OUTPUT/DIR'
entire_class_txt='./pnt_class_55.txt'
symbol_map_pair_json='./pointsymbols_pair_training.json'


def get_dict_from_txt(target_class_txt):
    f=open(target_class_txt,'r')
    lines=f.readlines()
    res_dict={}
    entire_list=[]
    for line in lines:
        kv=line.split(',')
        k=kv[0]
        v=kv[1]
        v=v.replace('\n','')
        res_dict[k]=v
        entire_list.append(k)
    print(res_dict)
    return res_dict

def get_pnt_name_from_mask(pnt_mask_pair,map_name,mask_name):
    output_pnt_name=''
    for pnt_list in pnt_mask_pair.items():
        pnt_name=pnt_list[0]
        img_list=pnt_list[1][0]
        mask_list=pnt_list[1][1]
        for img,mask in zip(img_list,mask_list):
            if map_name+'.tif'==img and mask_name==mask:
                output_pnt_name=pnt_name
                break
    if output_pnt_name!='':
        return output_pnt_name
    else:
        print(map_name,mask_name)
        print('error on retrieving the point name')
        return None


entire_list=os.listdir(input_root)
point_mask=[pnt for pnt in entire_list if pnt.endswith("_pt.tif")]
images=[img for img in entire_list if not img.endswith("_pt.tif") and not img.endswith("_poly.tif") and not img.endswith("_line.tif") and not img.endswith(".json")]

f=open(symbol_map_pair_json)
pair_json=json.load(f)

class_dict=get_dict_from_txt(entire_class_txt)

for img in images:
    img_path=os.path.join(input_root,img)
    map_name = os.path.basename(img_path).split('.')[0] # get the map name without extension

    image_output_dir = os.path.join(output_root,'images/')
    mask_output_dir = os.path.join(output_root,'annotation/')

    if not os.path.isdir(image_output_dir):
        os.makedirs(image_output_dir)
    if not os.path.isdir(mask_output_dir):
        os.makedirs(mask_output_dir)

    target_points=[pnt for pnt in point_mask if img.split('.')[0] in pnt.split('.')[0] and 'MF2364' not in pnt.split('.')[0]]

    shift_size = 1024

    map_img = Image.open(img_path)
    height, width = map_img.size 

    num_tiles_w = int(np.ceil(1. * width / shift_size))
    num_tiles_h = int(np.ceil(1. * height / shift_size))
    enlarged_width = int(shift_size * num_tiles_w)
    enlarged_height = int(shift_size * num_tiles_h)

    enlarged_map = Image.new(mode="RGB", size=(enlarged_width, enlarged_height))
    enlarged_map.paste(map_img) 


    for pnt in target_points:
        mask_path=os.path.join(input_root,pnt) 
        mask=Image.open(mask_path)
        mask=np.array(mask)
        idxs = np.where(mask == 1)
        save_img_from_this=[]  
        bb_list=[]
        tmp_list=[]
        point_name=pnt.split('.')[0].replace(map_name,'')
        point_name=point_name[1:]

        point_name=get_pnt_name_from_mask(pair_json,map_name,point_name)

        for ct_x,ct_y in zip(list(idxs)[0],list(idxs)[1]):
            top_left_x=ct_x-30
            top_left_y=ct_y-30
            bottom_right_x=ct_x+30
            bottom_right_y=ct_y+30
            tmp_list.append([top_left_y,top_left_x])
            tmp_list.append([bottom_right_y,bottom_right_x])
            bb_list.append(tmp_list)
            tmp_list=[]
    
        for idx in range(0, num_tiles_h):
            for jdx in range(0, num_tiles_w):
                for coord in bb_list:
                    flag=0
                    one_x_min=coord[0][0]
                    one_y_min=coord[0][1]
                    thr_x_max=coord[1][0]
                    thr_y_max=coord[1][1]
                    #other two coords in the BB 
                    two_x=thr_x_max
                    two_y=one_y_min
                    four_x=one_x_min
                    four_y=thr_y_max

                    center_x=one_x_min+30
                    center_y=one_y_min+30
                    if ((center_x >= jdx * shift_size) and (center_y >= idx * shift_size) and (center_x <= (jdx + 1) * shift_size) and (center_y <= (idx + 1) * shift_size)):        
                        if (one_x_min >= jdx * shift_size) and (one_y_min >= idx * shift_size) and (thr_x_max <= (jdx + 1) * shift_size) and (thr_y_max <= (idx + 1) * shift_size) :                       
                            bb_top_left_x=coord[0][0]-jdx * shift_size
                            bb_top_left_y=coord[0][1]-idx * shift_size
                            bb_bot_right_x=coord[1][0]-jdx * shift_size
                            bb_bot_right_y=coord[1][1]-idx * shift_size
                            flag=1

                        # upper x axis- if two coordinates included in the cropped region
                        elif ((one_x_min >= jdx * shift_size) and (one_y_min >= idx * shift_size) and (one_x_min <= (jdx + 1) * shift_size) and (one_y_min <= (idx + 1) * shift_size)) and ((two_x >= jdx * shift_size) and (two_y >= idx * shift_size) and (two_x <= (jdx + 1) * shift_size) and (two_y <= (idx + 1) * shift_size)) :
                            bb_top_left_x=one_x_min-jdx * shift_size
                            bb_top_left_y=one_y_min-idx * shift_size
                            bb_bot_right_x=two_x-jdx * shift_size
                            bb_bot_right_y=shift_size
                            flag=1

                        # lower x axis- if two coordinates included in the cropped region
                        elif ((thr_x_max >= jdx * shift_size) and (thr_y_max >= idx * shift_size) and (thr_x_max <= (jdx + 1) * shift_size) and (thr_y_max <= (idx + 1) * shift_size)) and ((four_x >= jdx * shift_size) and (four_y >= idx * shift_size) and (four_x <= (jdx + 1) * shift_size) and (four_y <= (idx + 1) * shift_size)) :
                            bb_top_left_x=four_x-jdx * shift_size
                            bb_top_left_y=0
                            bb_bot_right_x=thr_x_max-jdx * shift_size
                            bb_bot_right_y=thr_y_max-idx * shift_size
                            flag=1


                        # left y axis- if two coordinates included in the cropped region
                        elif ((thr_x_max >= jdx * shift_size) and (thr_y_max >= idx * shift_size) and (thr_x_max <= (jdx + 1) * shift_size) and (thr_y_max <= (idx + 1) * shift_size)) and ((two_x >= jdx * shift_size) and (two_y >= idx * shift_size) and (two_x <= (jdx + 1) * shift_size) and (two_y <= (idx + 1) * shift_size)) :
                            bb_top_left_x=0 
                            bb_top_left_y=two_y-idx * shift_size
                            bb_bot_right_x=thr_x_max-jdx * shift_size
                            bb_bot_right_y=thr_y_max-idx * shift_size
                            flag=1
                    
                        #right y axis- if two coordinates included in the cropped region                 
                        elif ((one_x_min >= jdx * shift_size) and (one_y_min >= idx * shift_size) and (one_x_min <= (jdx + 1) * shift_size) and (one_y_min <= (idx + 1) * shift_size)) and ((four_x >= jdx * shift_size) and (four_y >= idx * shift_size) and (four_x <= (jdx + 1) * shift_size) and (four_y <= (idx + 1) * shift_size)) :
                            bb_top_left_x=one_x_min-jdx * shift_size 
                            bb_top_left_y=one_y_min-idx * shift_size
                            bb_bot_right_x=shift_size
                            bb_bot_right_y=four_y-idx * shift_size
                            flag=1
                        #if only one coordinte included in the cropped region
                        elif ((one_x_min >= jdx * shift_size) and (one_y_min >= idx * shift_size) and (one_x_min <= (jdx + 1) * shift_size) and (one_y_min <= (idx + 1) * shift_size)):
                            bb_top_left_x=one_x_min-jdx * shift_size 
                            bb_top_left_y=one_y_min-idx * shift_size
                            bb_bot_right_x=shift_size
                            bb_bot_right_y=shift_size
                            flag=1
                        elif ((thr_x_max >= jdx * shift_size) and (thr_y_max >= idx * shift_size) and (thr_x_max <= (jdx + 1) * shift_size) and (thr_y_max <= (idx + 1) * shift_size)):
                            bb_top_left_x=0 
                            bb_top_left_y=0
                            bb_bot_right_x=thr_x_max-jdx * shift_size
                            bb_bot_right_y=thr_y_max-idx * shift_size
                            flag=1
                        elif ((four_x >= jdx * shift_size) and (four_y >= idx * shift_size) and (four_x <= (jdx + 1) * shift_size) and (four_y <= (idx + 1) * shift_size)):
                            bb_top_left_x=four_x-jdx * shift_size 
                            bb_top_left_y=0
                            bb_bot_right_x=shift_size
                            bb_bot_right_y=four_y-idx * shift_size
                            flag=1
                        elif ((two_x >= jdx * shift_size) and (two_y >= idx * shift_size) and (two_x <= (jdx + 1) * shift_size) and (two_y <= (idx + 1) * shift_size)):
                            bb_top_left_x=0 
                            bb_top_left_y=two_y-idx * shift_size
                            bb_bot_right_x=two_x-jdx * shift_size
                            bb_bot_right_y=shift_size
                            flag=1

                        if flag==1:
                            print(jdx * shift_size,idx * shift_size)
                            print(bb_top_left_x,bb_top_left_y,bb_bot_right_x,bb_bot_right_y)
                            anno_path=os.path.join(mask_output_dir,str(map_name)+'_h' + str(idx) + '_w' + str(jdx) + '.txt')
                            f = open(anno_path, "a")
                            f.write(str(class_dict[point_name]) + ',' + str(bb_top_left_x) + ',' + str(bb_top_left_y)+ ',' + str(bb_bot_right_x)+ ','+ str(bb_bot_right_y))
                            f.write('\n')
    #                         f.close()
                            img_clip = enlarged_map.crop((jdx * shift_size, idx * shift_size,(jdx + 1) * shift_size, (idx + 1) * shift_size, ))
                            img_out_path = os.path.join(image_output_dir,str(map_name)+ '_h' + str(idx) + '_w' + str(jdx) + '.jpg')
                            img_clip.save(img_out_path) 
