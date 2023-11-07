import glob
import os
import random
import shutil
import json
'''
Convert the annotation format with the yolo format
input file format : class, top-left-x, top-left-y, bottom-rigth-x,bottom-rigth-y

output file format : new_class, x_center , y_center, width, height
Each row is class x_center y_center width height format.
Box coordinates must be normalized by the dimensions of the image (i.e. have values between 0 and 1)
'''

entire_class_category=['arrow', 'arrow_circle', 'arrow_num', 'asterix', 'barbeque_tofu', 'barbeque_tofu_hollow', 'blue_arrow_kite', 'button', 'button2', 'c_dot', 'christmas_tree', 'christmas_jingle_bell', 'circle ', 'circle_info', 'crossed_downward_arrows', 'diamond_words', 'dot', 'fan', 'fault_line_triangle_hollow_num', 'fault_line_triangle_num', 'gear', 'gear_2', 'hollow_arrow', 'inclined_cleavage', 'inclined_line_box_hollow_num', 'inclined_line_box_num', 'line_diamond_center', 'line_diamond_center_solid', 'misc1', 'misc2', 'misc4', 'misc5', 'misc6', 'misc7', 'misc8', 'plus', 'plus2', 'plus_lollipop', 'purple_arrow_kite', 'quarry_open_pit', 'quarry_open_pit_U', 'reverse_p_num', 'reverse_p_num_dot', 'sleeping_y', 'small_inclined_fault_num', 'small_inclined_fault_num_2', 'small_inclined_fault_num_dot', 'solid_colored_circle', 'target', 'triangle', 'triangular_matrix', 'vertical_cleavage', 'vertical_cleavage_num', 'weight_lift', 'x']

#sample targeted symbols list
target_classes_list = [['button']['quarry_open_pit']]

global_img_w=1024
global_img_h=1024

input_img_root='./INPUT/DIR'
input_label_root='./INPUT_LABEL/DIR'

output_root='./OUTPUT/DIR'
output_train_val=os.path.join(output_root,'train_val/')
if not os.path.isdir(output_train_val):
    os.mkdir(output_train_val)

entire_class_path='./pnt_class_55.txt'
entire_class_dict=get_dict_from_txt(entire_class_path)

def convert_label_format(txt_file_path):
    img_txt_list=os.listdir(txt_file_path)
    for each_file in img_txt_list:
        if each_file.endswith('.txt'):
            # print(os.path.join(txt_file_path,each_file))
            with open(os.path.join(txt_file_path,each_file),'r') as f:
                lines=f.read()
                updated_contents = lines.replace(',',' ')           

            with open(os.path.join(txt_file_path,each_file),'w') as f:
                f.write(updated_contents) 
            

def gen_dict_from_class_list(target_class_list):
    target_dict={}
    num=0
    for class_name in target_class_list:
        target_dict[class_name]=num
        num+=1
    return target_dict

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
    # print(entire_list)
    return res_dict
def synth_filer_dataset_target_only(each_anno,img_w=global_img_w,img_h=global_img_h):

    new_annos_per_anno=[]

    synth_bb=each_anno["poly"]
    fit_min_x=synth_bb[0]
    fit_min_y=synth_bb[1]
    fit_max_x=synth_bb[4]
    fit_max_y=synth_bb[5]
    #resizing the BB from Synthmap
    center_x=(fit_min_x+fit_max_x)/2
    center_y=(fit_min_y+fit_max_y)/2
    min_x=center_x-30
    min_y=center_y-30
    max_x=center_x+30
    max_y=center_y+30
    
    x,y,w,h=convert_to_yolo(img_w,img_h, (int)(min_x),(int)(min_y),(int)(max_x),(int)(max_y))
    #only for one class
    new_annos_per_anno.append('0')
    new_annos_per_anno.append(x)
    new_annos_per_anno.append(y)
    new_annos_per_anno.append(w)
    new_annos_per_anno.append(h)

    return new_annos_per_anno
        


# def entire_class_category_dict(entire_class_path,target_class_list)
def filer_dataset_target_only(image_path,label_path,entire_class_dict,target_class_dict,output_path,img_w=global_img_w,img_h=global_img_h):
    '''
    1. only copy the annotation which are included in the targeted classes
    2. if the annotaion file is emtpy, we don't make a new file for this (also for the image pair too)
    3. make a new file: replace the class idx from entire to traget one 
    4. make a new file: convert from (minx miny maxx maxy) to (x,y,w,h) with normalized value (0 to 1)
    5. copy an image file with same file name 
    '''
    old_label=open(label_path,'r')
    old_lines=old_label.readlines()
    new_annos_per_file=[]
    check_targeted_set=[]
    for old_line in old_lines:
        tmp=[]
        each_anno_line=old_line.replace('\n','').split(',')
        class_idx=int(each_anno_line[0]) 
        # print(class_idx)     
        class_key=replace_new_class_key(entire_class_dict,class_idx)
        if class_key in target_class_dict.keys():
            # print(class_key)
            tmp.append(target_class_dict[class_key])
            x,y,w,h=convert_to_yolo(img_w,img_h, (int)(each_anno_line[1]),(int)(each_anno_line[2]),(int)(each_anno_line[3]),(int)(each_anno_line[4]))
            tmp.append(x)
            tmp.append(y)
            tmp.append(w)
            tmp.append(h)
            # print(tmp)
            new_annos_per_file.append(tmp)
            # if class_key not in check_targeted_set:
            #     check_targeted_set.append(class_key)
    # print(check_targeted_set)
    if len(new_annos_per_file)!=0:
        new_base_name=os.path.basename(label_path).split('.')[0]
        new_label_path=os.path.join(output_path,'image/',new_base_name+'.txt')    
        output_res=os.path.join(output_path,'image/')
        if not os.path.isdir(output_res):
            os.mkdir(output_res)
        # # make a new txt file
        new_f=open(new_label_path,'a')
        for line in new_annos_per_file:
            new_f.write(str(line[0])+' '+str(line[1])+' '+str(line[2])+' '+str(line[3])+' '+str(line[4]))
            new_f.write('\n')
        # # copy image file 
        new_img_path=os.path.join(output_res,new_base_name+'.jpg')
        shutil.copy(image_path,new_img_path)

def get_each_instances(data_path,target_classes):
    img_txt_list=os.listdir(data_path)
    each_instance={}
    for txt in img_txt_list:
        if txt.endswith('.txt'):
            f=open(os.path.join(data_path,txt),'r')
            lines=f.readlines()
            for line in lines:
                each_anno=line.split(' ')
                each_anno[0]=int(each_anno[0])
                if each_anno[0] not in each_instance.keys():
                    each_instance[each_anno[0]]=1
                else:
                    each_instance[each_anno[0]]+=1
    res_dict={}
    for key in target_classes.keys():
        for val in each_instance.keys():
            if target_classes[key] == val:
                if key not in res_dict.keys():
                    res_dict[key]=each_instance[val]
                else:
                    print('problem')

    print(res_dict)
def replace_new_class_key(entire_dict,old_class_idx):
    
    for key in entire_dict.keys():
        if int(entire_dict[key])==old_class_idx:
            class_key= key
    # print(class_key)
    return class_key

def convert_to_yolo(img_w,img_h, xmin,ymin,xmax,ymax):
    dw = 1./(img_w)
    dh = 1./(img_h)
    x = (xmin + xmax)/2.0 
    y = (ymin + ymax)/2.0
    # x = (xmin + xmax)/2.0 - 1
    # y = (ymin + ymax)/2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def split_train_valid(image_fold_path,label_fold_path,output_root):
    label_image_list=os.listdir(image_fold_path) 
    image_list=[]
    for img in label_image_list:
        if img.endswith('.jpg'):
            image_list.append(img)

    random.shuffle(image_list)
    print(len(image_list))
    train_size=(int)((0.90)*len(image_list))
    train_img=image_list[:train_size]
    val_img=image_list[train_size:]
    train_label=[]
    val_label=[] 
    for each_img in image_list:
        if each_img in val_img:
            new_output_root=os.path.join(output_root,'val/')
        else:
            new_output_root=os.path.join(output_root,'train/')       
        img_file_name=each_img
        mask_file_name=each_img.split('.')[0]+'.txt'
        if not os.path.isdir(new_output_root):
            os.mkdir(new_output_root)
        result_path=os.path.join(new_output_root,'image/')
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        shutil.move(os.path.join(image_fold_path,img_file_name),os.path.join(result_path,img_file_name))
        shutil.move(os.path.join(label_fold_path,mask_file_name),os.path.join(result_path,mask_file_name))
 
def move_to_one_dir(dir_path,out_path):
    dir_list=os.listdir(dir_path)
    for each_dir in dir_list:
        each_dir_path=os.path.join(dir_path,each_dir)
        each_file_list=os.listdir(each_dir_path)
        for each_file in each_file_list:
            each_file_path=os.path.join(each_dir_path,each_file)
            new_file_path=os.path.join(out_path,each_file)
            # print(each_file_path,new_file_path)
            shutil.move(each_file_path,new_file_path)




'''
using competition data as an input
'''

for target_classes in target_classes_list:
    target_class_dict=gen_dict_from_class_list(target_classes)
    output_dir_name=target_classes[0]
    output_train_val_path=os.path.join(output_train_val,output_dir_name)
    print(output_train_val_path)
    if not os.path.isdir(output_train_val_path):
        os.mkdir(output_train_val_path)
    label_list=os.listdir(input_label_root)
    label_list.sort()
    img_list=os.listdir(input_img_root)
    img_list.sort()
    for old_label,old_img in zip(label_list,img_list):       
        if old_label.split('.')[0] != old_img.split('.')[0]:
            print(old_label,old_img)
        old_label_path=os.path.join(input_label_root,old_label)
        old_img_path=os.path.join(input_img_root,old_img)
        filer_dataset_target_only(old_img_path,old_label_path,entire_class_dict,target_class_dict,output_train_val_path)

for target_classes in target_classes_list:
    output_dir_name=target_classes[0]
    output_train_val_path=os.path.join(output_train_val,output_dir_name)
    split_train_valid(os.path.join(output_train_val_path,'image/'),os.path.join(output_train_val_path,'image/'),output_train_val_path)


