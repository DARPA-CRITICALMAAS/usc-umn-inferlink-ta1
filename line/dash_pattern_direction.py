import os
import cv2
import numpy as np
from helper.process_shp import read_shp
from shapely.geometry import LineString
from PIL import Image
from line_direction import extract_template_from_legend, rotate_image, calculate_line_slop, template_match, check_direction

def generate_dash_roi_binary_image(map_image, line_mask_, line, seg_line_shp, line_width=15):
    xmin, ymin, xmax, ymax = seg_line_shp.bounds
    xmin, ymin, xmax, ymax = int(xmin)-10, int(ymin)-10, int(xmax)+10, int(ymax)+10       

    map_subregion = map_image[xmin: xmax, ymin:ymax]
    
    map_subregion_pil = Image.fromarray(map_subregion)
    r, g, b = map_subregion_pil.split()
    r = r.point(lambda p: p + 1.1)
    g = g.point(lambda p: p + 1.1)
    b = b.point(lambda p: p + 1.1)
    map_subregion_pil = Image.merge("RGB", (r, g, b))
    map_subregion = np.asarray(map_subregion_pil)
    
#     line_mask_ = np.ones(map_image.shape) * 255
    line_mask_ = draw_line_in_image(line, line_mask_, line_width)
    line_mask = line_mask_[xmin:xmax, ymin:ymax]
    line_area = (map_subregion * line_mask)
    
    avg_gray_val = np.average(line_area[np.where(line_area<255)])
    line_area[(line_area > 255)] = 255
    line_area = line_area.astype('uint8')
    line_area_gray = cv2.cvtColor(line_area, cv2.COLOR_BGR2GRAY) # white is bg
    min_gray_val = np.min(line_area_gray)
    
    binary_thres = (avg_gray_val + min_gray_val) // 2
    
    _, binary_image = cv2.threshold(line_area_gray, binary_thres, 255, cv2.THRESH_BINARY_INV)
    return map_subregion, line_area_gray, binary_image

def generate_direction_roi_binary_image(map_image, line_mask_, line, seg_line_shp, line_width=50):
    xmin, ymin, xmax, ymax = seg_line_shp.bounds
    xmin, ymin, xmax, ymax = int(xmin)-10, int(ymin)-10, int(xmax)+10, int(ymax)+10       
    width = max(xmax-xmin, ymax-ymin)+ 10
    map_subregion = map_image[xmin:xmin+width, ymin:ymin+width] 

    map_subregion_pil = Image.fromarray(map_subregion)
    r, g, b = map_subregion_pil.split()
    r = r.point(lambda p: p + 1.1)
    g = g.point(lambda p: p + 1.1)
    b = b.point(lambda p: p + 1.1)
    map_subregion_pil = Image.merge("RGB", (r, g, b))
    map_subregion = np.asarray(map_subregion_pil)
    
#     line_mask_ = np.ones(map_image.shape) * 255
    line_mask_ = draw_line_in_image(line, line_mask_, line_width)
    line_mask = line_mask_[xmin:xmin+width, ymin:ymin+width]
    line_area = (map_subregion * line_mask)

    avg_gray_val = np.average(line_area[np.where(line_area<255)])
    line_area[(line_area > 255)] = 255
    line_area = line_area.astype('uint8')
    line_area_gray = cv2.cvtColor(line_area, cv2.COLOR_BGR2GRAY) # white is bg
    min_gray_val = np.min(line_area_gray)

    binary_thres = (avg_gray_val + min_gray_val) // 2
    _, binary_image = cv2.threshold(line_area_gray, binary_thres, 255, cv2.THRESH_BINARY_INV)
    
    seg_line_coords = seg_line_shp.coords
    scale = np.array([xmin, ymin])
    seg_line_scaled = seg_line_coords - scale
    
    return map_subregion, line_area, binary_image, seg_line_scaled

def draw_line_in_image(line_list, empty_image, line_width):
    for i in range(1, len(line_list)):
        pt1 = (int(line_list[i-1][1]), int(line_list[i-1][0]))
        pt2 = (int(line_list[i][1]), int(line_list[i][0]))
        cv2.line(empty_image, pt1, pt2, (1,1,1), line_width)
    return empty_image

def interpolate_line(line, segment_length):
    split_points = []
    split_points.append(line.interpolate(segment_length))
    interpolated_line = list(line.coords)[:-1] \
                                   + [(int(point.x), int(point.y)) for point in split_points] \
                                   + [list(line.coords)[-1]]
    line_segments = []
    for i in range(1, len(interpolated_line)):
        line_segments += [interpolated_line[i-1], interpolated_line[i]]
    return line_segments
    
def split_line(line, segment_length):
    segmented_line_list = []
    total_length = line.length
    line_list = list(line.coords)
    
    line_interp_list = [line_list[0]]
    cur_line = [line_list[0]]
    for cur_pt in line_list[1:]:
        cur_line.append(cur_pt)
        temp_line = LineString(cur_line)
        if temp_line.length - segment_length > 10.0:
            temp_line_interp = interpolate_line(temp_line, 50)
            line_interp_list += temp_line_interp
            cur_line = [cur_pt]
        else:
            line_interp_list.append(cur_pt)

    cur_line = [line_interp_list[0]]
    for cur_pt in line_interp_list[1:]:
        cur_line.append(cur_pt)
        temp_line = LineString(cur_line)
        if abs(temp_line.length - segment_length) < 15.0:
            segmented_line_list.append(cur_line)
            cur_line = [cur_pt]
        elif temp_line.length - segment_length > 10.0:
            segmented_line_list.append(cur_line)
            cur_line = [cur_pt]
        elif cur_pt == line_interp_list[-1]:
            segmented_line_list.append(cur_line)
    return segmented_line_list
        
def categorize_dash_pattern(cc_sizes):
    refined_cc_sizes = [i for i in cc_sizes if i > 5 ]
    if len(refined_cc_sizes) == 0:
        return 'unknown'
    avg = sum(refined_cc_sizes) / float(len(refined_cc_sizes))
    if max(refined_cc_sizes) > 200 or len(refined_cc_sizes) == 1:
        return 'solid'
    elif abs(avg - 20) < 10 or len(refined_cc_sizes) >= 4:
        return 'dotted'
    elif abs(avg - 50) < 10 or len(refined_cc_sizes) < 4:
        return 'dash'
    else:
        return 'unknown'

def detect_line_dash_direction(shapefile_path, map_png_path, map_tif_dir, \
                               obj_name='normal_fault_line', match_threshold=0.7):
    polylines = read_shp(shapefile_path)

    map_image = cv2.imread(map_png_path)
    line_mask_ = np.ones(map_image.shape) * 255
    
    map_name = map_png_path.split('/')[-1][:-4]
    
    symbol_template = extract_template_from_legend(map_name, obj_name=obj_name, map_dir=map_tif_dir)
    
    print('number of lines in the original shapefile: ', len(polylines))
    print('map shape: ', map_image.shape)

    line_dict = {}
    
    for idx, line in enumerate(polylines):
        line_shp = LineString(line)
        length = line_shp.length

        segmented_lines = split_line(line_shp, 50)
            
        for i, seg_line in enumerate(segmented_lines):
            seg_line_shp = LineString(seg_line)
            line_dict[str(seg_line_shp)] = []
            
            sub_map_image, line_roi, bin_line_roi = generate_dash_roi_binary_image(map_image, line_mask_, line, seg_line_shp)
#             cv2.imwrite('./test_res/{}_{}_raw.png'.format(idx, i), sub_map_image)
#             cv2.imwrite('./test_res/{}_{}_roi.png'.format(idx, i), line_roi)
#             cv2.imwrite('./test_res/{}_{}_bin.png'.format(idx, i), bin_line_roi)
            ##################################
            # dash pattern
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_line_roi, connectivity=8)
            sizes = stats[1:, -1]        

            dash_pattern = categorize_dash_pattern(sizes)
            line_dict[str(seg_line_shp)].append(dash_pattern)
#             line_dict[str(seg_line_shp)].append('solid')
            ##################################
            
            ##################################
            # direction
            if symbol_template is None:
                print('--- no symbol ---')
                line_dict[str(seg_line_shp)].append(0)
            else:
                sub_map_image, line_roi, bin_line_roi, scaled_line_np = \
                                                generate_direction_roi_binary_image(map_image, line_mask_, line, seg_line_shp)

                line_slope = calculate_line_slop(seg_line_shp)

                rotated_template_list = []
                for angle in [0, 90, 180, 270]:         
                    rot_template = rotate_image(symbol_template, line_slope+angle, output_path=None)
                    rotated_template_list.append(np.array(rot_template).astype('uint8'))
                dire = 0
                for i, rot_temp in enumerate(rotated_template_list):
                    symbol_center = template_match(bin_line_roi, rot_temp, threshold=match_threshold)
                    if symbol_center:
                        print('detected')
                        break
                if symbol_center:
                    dire, _,  _ = check_direction(scaled_line_np[0], scaled_line_np[-1], symbol_center)
                    line_dict[str(seg_line_shp)].append(dire)
    #                 cv2.imwrite('./test_res/{}_{}_raw.png'.format(idx, i), sub_map_image)
    #                 cv2.imwrite('./test_res/{}_{}_roi.png'.format(idx, i), line_roi)
    #                 cv2.imwrite('./test_res/{}_{}_bin.png'.format(idx, i), bin_line_roi)
                else:
                    line_dict[str(seg_line_shp)].append(0)
            line_mask_[:] = 255
    #             if dash_pattern == 'unknown':
    #                 cv2.imwrite('./test_res/{}_{}_raw.png'.format(idx, i), sub_map_image)
    #                 cv2.imwrite('./test_res/{}_{}_roi.png'.format(idx, i), line_roi)
    #                 cv2.imwrite('./test_res/{}_{}_bin.png'.format(idx, i), bin_line_roi)
    #                 print(f"{str(seg_line_shp)}: {dash_pattern}, {dire}")
    return line_dict



        