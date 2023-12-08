import os
import cv2
import numpy as np
from helper.process_shp import read_shp
from shapely.geometry import LineString
from PIL import Image
import math
import json

def draw_line_in_image(line_list, empty_image, buffer=50):
    for i in range(1, len(line_list)):
        pt1 = (line_list[i-1][0], line_list[i-1][1])
        pt2 = (line_list[i][0], line_list[i][1])
        cv2.line(empty_image, pt1, pt2, (1,1,1), buffer)
    return empty_image

def interpolate_line(line, segment_length):
    split_points = []
    split_points.append(line.interpolate(segment_length))
    interpolated_line = list(line.coords)[:-1] \
                                   + [(point.x, point.y) for point in split_points] \
                                   + [list(line.coords)[-1]]
    line_segments = []
    for i in range(1, len(interpolated_line)):
        line_segments += [interpolated_line[i-1], interpolated_line[i]]
    return line_segments

def rotate_image(image_np, angle, output_path=None):
    image = Image.fromarray(image_np)    
    # Rotate the image
    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)
    # Convert the image to grayscale
    grayscale_image = rotated_image.convert('L')
    # Convert the grayscale image to binary using a threshold
    binary_image = grayscale_image.point(lambda x: 0 if x < 30 else 255, '1')
    # Save or display the rotated image
    if output_path:
        rotated_image.save(output_path)
    return binary_image

def slope2angle(slope):
    # Calculate the angle in radians
    radian_angle = math.atan(slope)
    
    # Convert the angle to degrees
    degree_angle = math.degrees(radian_angle)
    
    # Convert the angle to an integer
    int_angle = int(round(degree_angle))
    
    return int_angle

def template_match(main_image, template, threshold=0.7):
    # Get the width and height of the template
    w, h = template.shape[::-1]
    w_m, h_m = main_image.shape[::-1]
    
    if w_m < w or h_m <h:
#         print('template is larger', (w, h), (w_m, h_m))
        return None
    
    # Perform template matching
    result = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)
    
    loc = np.where(result >= threshold)
    if loc[0].shape[0] == 0:
        return None
    for pt in zip(*loc[::-1]):
        return [pt[1] + h//2, pt[0] + w//2]

def project_pt(pt1, pt2, pt):
    # Define the points A, B, and P as NumPy arrays
    A = np.array([pt1[0], pt1[1]])
    B = np.array([pt2[0], pt2[1]])
    P = np.array([pt[0], pt[1]])
    # Step 1: Calculate vector AP
    AP = P - A
    # Step 2: Calculate vector AB
    AB = B - A
    # Step 3: Calculate the projection of AP onto AB
    # This is done by finding the dot product of AP and AB, divided by the dot product of AB with itself
    # Then multiply the unit vector AB by this scalar
    projection_scalar = np.dot(AP, AB) / np.dot(AB, AB)
    projection_vector = projection_scalar * AB
    # Step 4: Find the coordinates of the projection
    projection_point = A + projection_vector
    return (projection_point)

def project_pt(pt1, pt2, pt):
    # Define the points A, B, and P as NumPy arrays
    A = np.array([pt1[0], pt1[1]])
    B = np.array([pt2[0], pt2[1]])
    P = np.array([pt[0], pt[1]])
    # Step 1: Calculate vector AP
    AP = P - A
    # Step 2: Calculate vector AB
    AB = B - A
    # Step 3: Calculate the projection of AP onto AB
    # This is done by finding the dot product of AP and AB, divided by the dot product of AB with itself
    # Then multiply the unit vector AB by this scalar
    projection_scalar = np.dot(AP, AB) / np.dot(AB, AB)
    projection_vector = projection_scalar * AB
    # Step 4: Find the coordinates of the projection
    projection_point = A + projection_vector
    return (projection_point)

def check_direction(line_pt1, line_pt2, symbol_pt):
    projected_pt = project_pt(line_pt1, line_pt2, symbol_pt)
    proj_symbol_pt = (symbol_pt[0] - projected_pt[0]) * (symbol_pt[1] - projected_pt[1])
    if proj_symbol_pt < 0:
        return -1, projected_pt, proj_symbol_pt
    elif proj_symbol_pt > 0:
        return 1, projected_pt, proj_symbol_pt
    else:
        return 0, projected_pt, proj_symbol_pt

def extract_template_from_legend(map_name, obj_name='fault_line', symbol_height=10, symbol_width=20, stride=10, \
                                map_dir = '/data/weiweidu/criticalmaas_data/training',\
                                template_save_path = './symbol_template'):
    ############################################
    ## find the json file for the bbox of the obj
    ############################################
    bbox = []
    
    # check bbox for normal fault line
    json_path = os.path.join(map_dir, map_name+'.json')
    f = open(json_path)
    metadata = json.load(f)
    for symbol in metadata['shapes']:
        if symbol['label'].lower() == obj_name:
            bbox = symbol['points']

    print(f'--- Get the bounding box of {obj_name} on {map_name} ---')
    
    # check bbox for fault line
    if bbox == []:
        for symbol in metadata['shapes']:
            if symbol['label'].lower() == 'fault_line':
                bbox = symbol['points']
    # if not bbox for symbol, return None (no template)
    if bbox == []:
        return None
    
    map_path = os.path.join(map_dir, map_name+'.tif')
    map_image = cv2.imread(map_path)
    
    line_legend = map_image[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])]
    gray_line_legend = cv2.cvtColor(line_legend, cv2.COLOR_BGR2GRAY)
    _, bin_line_legend = cv2.threshold(gray_line_legend, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_line_legend, connectivity=8)

    # Display the number of connected components found
#     print(f"Number of connected components: {num_labels - 1}")  # Subtract 1 to exclude the background
    bbox = stats[1:, :]
    
    ## find the bbox with some height (not only a line)
    box_height = [box for box in bbox if box[3] > 10]
    
    if box_height == []:
        return None
    
    ## find the largest bbox, the symbol should in the largest bbox
    max_box = []
    max_area = 0
    for box in box_height:
        if box[-1] > max_area:
            max_box = box
            max_area = box[-1]
    
    y1, x1 = max_box[0], max_box[1]
    y2, x2 = y1 + max_box[2], x1 + max_box[3]
    w_, h_ = max_box[2], max_box[3]

    y_ = np.int32(np.linspace(y1+5, y2-20, w_//stride))

    symbol_box = []
    max_size = 0
    for y_s in y_:
        sub = bin_line_legend[x1:x1+h_, y_s:y_s+symbol_width]
        area = np.sum(sub/255)
        if area > max_size:
            max_size = area
            symbol_box = [x1, y_s, x1+h_, y_s+symbol_width]
    
    if symbol_box != []:
        symbol_image = bin_line_legend[symbol_box[0]:symbol_box[2], symbol_box[1]:symbol_box[3]]
        cv2.imwrite(f'{template_save_path}/{map_name}.png', symbol_image)
        print(f'--- saving symbol image in {template_save_path}/{map_name}.png ---')
    else:
        symbol_image = None
    return symbol_image

def generate_roi_binary_image(map_image, line, seg_line_shp):
    ymin, xmin, ymax, xmax = seg_line_shp.bounds
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
    line_mask_ = np.ones(map_image.shape) * 255
    line_mask_ = draw_line_in_image(line, line_mask_)
    line_mask = line_mask_[xmin:xmin+width, ymin:ymin+width]
    line_area = (map_subregion * line_mask)
    avg_gray_val = np.average(line_area[np.where(line_area<255)])
    line_area[(line_area > 255)] = 255
    line_area = line_area.astype('uint8')
    line_area_gray = cv2.cvtColor(line_area, cv2.COLOR_BGR2GRAY) # white is bg
    min_gray_val = np.min(line_area_gray)
    binary_thres = (avg_gray_val + min_gray_val) // 2
    _, binary_image = cv2.threshold(line_area_gray, binary_thres, 255, cv2.THRESH_BINARY_INV)
    return map_subregion, line_area, binary_image

def calculate_line_slop(seg_line_shp):
    rotate_angle = 0
    x1, y1 = seg_line_shp.coords[0]
    x2, y2 = seg_line_shp.coords[-1]

    if x1 == x2:
        rotate_angle = 90
    elif y1 == y2:
        rotate_angle = 180
    else:
        slope = (y2-y1) / (x2-x1)
        rotate_angle = slope2angle(slope)
    return rotate_angle
    
# def extract_symbol_along_line(map_name,\
#                               map_dir = '/data/weiweidu/criticalmaas_data/training_fault_line_comb',\
#                               legend_dir = '/data/weiweidu/criticalmaas_data/training',\
#                               roi_buffer=50, line_length=50, match_threshold=0.7):
    
#     shapefile_path = f'{map_dir}/{map_name}_fault_line.shp'
#     map_image_path = f'{map_dir}/{map_name}.png'

#     polylines = read_shp(shapefile_path)

#     map_image = cv2.imread(map_image_path)

#     print('number of lines in the original shapefile: ', len(polylines))
#     print('map shape: ', map_image.shape)
    
#     symbol_template = extract_template_from_legend(map_name, map_dir=legend_dir)

#     for idx, line in enumerate(polylines):
#         line_shp = LineString(line)
#         length = line_shp.length
# #         if 100 < length < 200:
#         segmented_lines = split_line(line_shp, line_length)
#         for i, seg_line in enumerate(segmented_lines):
# #             if i != 0:
# #                 continue
#             seg_line_shp = LineString(seg_line)
#             sub_map_image, line_roi, bin_line_roi = generate_roi_binary_image(map_image, line, seg_line_shp)
#             line_slope = calculate_line_slop(seg_line_shp)

#             rotated_template_list = []
#             for angle in [0, 90, 180, 270]:         
#                 rot_template = rotate_image(symbol_template, line_slope+angle, output_path=None)
#                 rotated_template_list.append(np.array(rot_template).astype('uint8'))

#             for i, rot_temp in enumerate(rotated_template_list):
#                 matched_res = template_match(bin_line_roi, rot_temp, threshold=match_threshold)
#                 if matched_res is not None:
#                     break
#             if matched_res is not None:
#                 print(f'matched {idx}_{i}')
#                 cv2.imwrite('./test_res/{}_{}_raw.png'.format(idx, i), sub_map_image)
#                 cv2.imwrite('./test_res/{}_{}_roi.png'.format(idx, i), line_roi)
#                 cv2.imwrite('./test_res/{}_{}_bin.png'.format(idx, i), bin_line_roi)
#                 cv2.imwrite('./test_res/{}_{}_mat.png'.format(idx, i), matched_res)
#         print('====================') 

if __name__ == '__main__':
    extract_symbol_along_line('CA_LosAngeles', \
                              map_dir = '/data/weiweidu/criticalmaas_data/training_fault_line_comb', \
                              legend_dir = '/data/weiweidu/criticalmaas_data/training',\
                              match_threshold=0.5)