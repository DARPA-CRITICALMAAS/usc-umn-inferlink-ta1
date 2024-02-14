import os
import cv2
import yaml
import json
from argparse import ArgumentParser
import pdb
import numpy as np
import collections
from shapely.ops import *
from shapely import geometry
from shapely.strtree import STRtree
from helper.process_shp import write_shp_in_imgcoord, rm_dup_lines, integrate_lines, write_shp_in_imgcoord_output_schema
from write_shp_schema import write_shp_in_imgcoord_with_attr
from line_ornament import extract_attributes_along_line

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--trained_model_dir', default=None, help='directory storing trained models')
parser.add_argument('--checkpoint', default=None, help='checkpoint of the model to test.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[3],
                        help='list of index where skip conn will be made.')
parser.add_argument('--predict_raster', action='store_true', help='predict results are png if True')
parser.add_argument('--predict_vector', action='store_true', help='predict results are geojson if True')
parser.add_argument('--line_feature_name', default='fault_line', type=str, help='the name of line feature')
parser.add_argument('--cropped_image_dir', type=str, default='/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines',
                       help='the path to the cropped images')
parser.add_argument('--map_legend_json', default=None, type=str, help='json from the map legend module')
parser.add_argument('--prediction_dir', type=str, default='./pred_maps',
                       help='the path to save prediction results')
parser.add_argument('--map_name', type=str, default=None)
parser.add_argument('--buffer', type=int, default=10,
                        help='the buffer size for nodes conflation')


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def add_lines(line2add, line_list, args):
    for i in range(len(line_list)):
        line = line_list[i]
        if line.distance(line2add) > 1:
            continue
        print('processing {}'.format(line2add))
        buffered_line = line.buffer(args.buffer)
        intersect = buffered_line.intersection(line2add)
        if not intersect:
            continue
        elif intersect.equals(line2add):
            print('duplicate')
            return line_list
        else:
            line_list[i] = line.union(line2add)
            print('union')
            return line_list
    line_list.append(line2add)
    return line_list

def add_lines_sindex(line2add, line_list, args):
    tree = STRtree(line_list)
    query_line = line2add.buffer(args.buffer)
    flag = False
    for o in tree.query(query_line):
        if o.intersects(query_line):
            intersect = o.intersection(query_line)
            o_n1, o_n2 = list(o.coords)
            o_n1, o_n2 = geometry.Point(o_n1), geometry.Point(o_n2)
            n12add, n22add = list(line2add.coords)
            n12add, n22add  = geometry.Point(n12add), geometry.Point(n22add)
            #highly overlapped/duplicated lines
            if o.buffer(args.buffer).contains(line2add): #(o_n1.distance(line2add) < 10 and o_n2.distance(line2add) < 10)
                return line_list  
            if query_line.contains(o):
                idx = line_list.index(o)
                del line_list[idx]
                flag = True
                break
                
    query_n1, query_n2 = list(line2add.coords)
    query_n1, query_n2 = geometry.Point(query_n1), geometry.Point(query_n2)
    buffered_query_n1, buffered_query_n2 = query_n1.buffer(args.buffer), query_n2.buffer(args.buffer)
    line4query_n1, line4query_n2 = None, None
    if flag:
        tree = STRtree(line_list)
    for o in tree.query(buffered_query_n1):
        if o.intersects(buffered_query_n1):
            line4query_n1 = o
            break
    for o in tree.query(buffered_query_n2):
        if o.intersects(buffered_query_n2):
            line4query_n2 = o
            break
    if line4query_n1 and line4query_n2:
        n11, n12 = list(line4query_n1.coords)
        n11, n12 = geometry.Point(n11), geometry.Point(n12)
        n21, n22 = list(line4query_n2.coords)
        n21, n22 = geometry.Point(n21), geometry.Point(n22)
        dist = np.array([geometry.LineString([n11, n21]).length, geometry.LineString([n11, n22]).length, \
                geometry.LineString([n12, n21]).length, geometry.LineString([n12, n22]).length])
        argmin = np.argmin(dist)
        if argmin == 0:
            line_list.append(geometry.LineString([n11, n21]))
        if argmin == 1:
            line_list.append(geometry.LineString([n11, n22]))    
        if argmin == 2:
            line_list.append(geometry.LineString([n12, n21]))    
        if argmin == 3:
            line_list.append(geometry.LineString([n12, n22]))  
        return line_list
    elif line4query_n1:
        n11, n12 = list(line4query_n1.coords)
        n11, n12 = geometry.Point(n11), geometry.Point(n12)
        dist = np.array([geometry.LineString([n11, query_n2]).length, geometry.LineString([n12, query_n2]).length])
        argmin = np.argmin(dist)
        if argmin == 0:
            line_list.append(geometry.LineString([n11, query_n2]))
        if argmin == 1:
            line_list.append(geometry.LineString([n12, query_n2]))
        return line_list
    elif line4query_n2:
        n21, n22 = list(line4query_n2.coords)
        n21, n22 = geometry.Point(n21), geometry.Point(n22)
        dist = np.array([geometry.LineString([n21, query_n1]).length, geometry.LineString([n22, query_n1]).length])
        argmin = np.argmin(dist)
        if argmin == 0:
            line_list.append(geometry.LineString([n21, query_n1]))
        if argmin == 1:
            line_list.append(geometry.LineString([n22, query_n1]))
        return line_list
        
    line_list.append(line2add)
    return line_list
            
    
def construct_graph(args, map_content_mask):    
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    
    config.DATA.TEST_DATA_PATH = args.cropped_image_dir
    config.DATA.PRED_MAP_NAME = args.map_name + f'_{args.line_feature_name}_pred'
    if 'thrust_fault_line_model.pt' in args.checkpoint:
        config.MODEL.DECODER.OBJ_TOKEN = 100
    elif 'fault_line_model.pt' in args.checkpoint:
        config.MODEL.DECODER.OBJ_TOKEN = 150
    
    import torch
    from monai.data import DataLoader
    from tqdm import tqdm
    import numpy as np

    from dataset_road_network import build_road_network_data
    from models import build_model
    from inference import relation_infer
    from metric_smd import StreetMoverDistance
    from metric_map import BBoxEvaluator
    from box_ops_2D import box_cxcywh_to_xyxy_np
    from utils import image_graph_collate_road_network

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")
    
    net = build_model(config).to(device)

    test_ds, img_names = build_road_network_data(
        config, mode='test'
    )

    test_loader = DataLoader(test_ds,
                            batch_size=config.DATA.TEST_BATCH_SIZE,
                            shuffle=False,
                            num_workers=config.DATA.NUM_WORKERS,
                            collate_fn=image_graph_collate_road_network,
                            pin_memory=True)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cnt = 0
    
    lines = [] # a list of LineString 
    
    with torch.no_grad():
        print('Started processing test set.')
        batch_cnt = 0
        for batchdata in tqdm(test_loader):
            # extract data and put to device
            images, segs, nodes, edges = batchdata[0], batchdata[1], batchdata[2], batchdata[3]
            segs_np = segs.cpu().numpy()
            images_np = images.permute(0, 2, 3, 1).cpu().numpy()
            images_np = ((images_np * std + mean) * 255).astype('int32')
#             print('gt shape', segs.shape)
            images = images.to(args.device,  non_blocking=False)
            segs = segs.to(args.device,  non_blocking=False)
            nodes = [node.to(args.device,  non_blocking=False) for node in nodes]
            edges = [edge.to(args.device,  non_blocking=False) for edge in edges]

            h, out, _, _, _, _ = net(images, seg=False)
            pred_nodes, pred_edges, _, pred_nodes_box, pred_nodes_box_score,\
            pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = \
            relation_infer(
                h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
                nms=False, map_=True)
#             print('pred_edges shape: ', pred_edges)
            img_size = config.DATA.IMG_SIZE[0]
            for cnt, val in enumerate(zip(pred_edges, pred_nodes)):
                indices = img_names[batch_cnt*config.DATA.TEST_BATCH_SIZE+cnt][:-4].split('_')
                x_id, y_id = int(indices[-2]), int(indices[-1])
                patch_content_mask = map_content_mask[x_id:x_id+img_size, y_id:y_id+img_size]
                if np.sum(patch_content_mask) < 50:
                    continue
                
                edges_, nodes_ = val
                nodes_ = nodes_.cpu().numpy()
                
                if edges_.shape[0] < 5:
                    continue
#                 print('positive  prediction')
                for i_idx, j_idx in edges_:                 
                    n1, n2 = (nodes_[i_idx]*img_size).astype('int32'), (nodes_[j_idx]*img_size).astype('int32')
                    indices = img_names[batch_cnt*config.DATA.TEST_BATCH_SIZE+cnt][:-4].split('_')
                    x_id, y_id = int(indices[-2]), int(indices[-1])

                    n1_in_map = [n1[0] + x_id, n1[1] + y_id]
                    n2_in_map = [n2[0] + x_id, n2[1] + y_id]
                    
                    line = geometry.LineString([[n1_in_map[0],n1_in_map[1]], [n2_in_map[0],n2_in_map[1]]])
                    lines = add_lines_sindex(line, lines, args)
                    
            batch_cnt += 1
    return lines

def predict_png(args):
    """
    generate png prediction result
    """
    # Load the config files
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)

    config.DATA.PRED_MAP_NAME = args.map_name + f'_{args.line_feature_name}_pred'
    
    map_name = args.map_name

    segment_json_file = open(os.path.join(args.map_legend_json, args.map_name+'_line.json'))
    segment_json = json.load(segment_json_file)
    
    map_content_y, map_content_x, map_content_w, map_content_h= segment_json['map_content_box']
    map_height, map_width = segment_json['map_dimension']
    
    map_content_mask = np.zeros((map_height, map_width))
    cv2.rectangle(map_content_mask, (int(map_content_y),int(map_content_x)), \
                  (int(map_content_y)+int(map_content_w),int(map_content_x)+int(map_content_h)), 255, -1)
    
    lines = construct_graph(args, map_content_mask)
 
    pred_png = np.zeros((map_height, map_width, 3))
    for line in lines:                       
        node1, node2 = list(line.coords)
        node1, node2 = [int(node1[0]), int(node1[1])], [int(node2[0]), int(node2[1])]
        cv2.line(pred_png, (node1[1], node1[0]), (node2[1], node2[0]), (255,255,255), 1)
    save_path = f'{args.prediction_dir}/{config.DATA.PRED_MAP_NAME}.png'
    cv2.imwrite(save_path, pred_png)
    print('*** save the predicted map in {} ***'.format(save_path))

def predict_shp(args):
    """
    generate shp prediction
    """
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
                                        
    config.DATA.PRED_MAP_NAME = args.map_name + f'_{args.line_feature_name}_pred'
    
    map_name = args.map_name
    
    segment_json_file = open(os.path.join(args.map_legend_json, args.map_name+'_line.json'))
    segment_json = json.load(segment_json_file)
    
    map_content_y, map_content_x, map_content_w, map_content_h= segment_json['map_content_box']
    map_height, map_width = segment_json['map_dimension']
    
    map_content_mask = np.zeros((map_height, map_width))
    cv2.rectangle(map_content_mask, (int(map_content_y),int(map_content_x)), \
                  (int(map_content_y)+int(map_content_w),int(map_content_x)+int(map_content_h)), 255, -1)
                                         
    lines = construct_graph(args, map_content_mask)
    nodup_lines = rm_dup_lines(lines)
    print('num of lines = ', len(nodup_lines))
    
    if not os.path.exists(args.prediction_dir):
        os.mkdir(args.prediction_dir)
    
    shp_path = f'{args.prediction_dir}/{config.DATA.PRED_MAP_NAME}.shp'
    if len(nodup_lines) > 0:
        merged_lines = integrate_lines(nodup_lines)
        write_shp_in_imgcoord_output_schema(shp_path, merged_lines)
        print('*** save the predicted shapefile in {} ***'.format(shp_path))
    else:
        write_shp_in_imgcoord_output_schema(shp_path, nodup_lines)
        
    import geopandas
    geojson_output_dir = f'{args.prediction_dir}/{args.map_name}'
    if not os.path.exists(geojson_output_dir):
        os.mkdir(geojson_output_dir)
        
    geojson_path = f'{args.prediction_dir}/{args.map_name}/{config.DATA.PRED_MAP_NAME}.geojson'
    shp_file = geopandas.read_file(shp_path)
    shp_file.to_file(geojson_path, driver='GeoJSON')
    print('*** save the predicted geojson in {} ***'.format(geojson_path))
    return shp_path
    

if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.map_legend_json, args.map_name+'_line.json')
    f = open(json_path)
    data = json.load(f)
    
    for _, sym_property in data.items():
        if not isinstance(sym_property, dict):
            continue
        sym_name = sym_property['symbol name']
        description = sym_property['description']
        if 'fault' in description.lower() or 'fault' in sym_name.lower():            
            args.checkpoint = f'{args.trained_model_dir}/fault_line_model.pt'
            args.line_feature_name = 'fault_line'
            if args.predict_raster:
                predict_png(args)
            if args.predict_vector:
                output_shp_path = predict_shp(args)
#                 line_dict = extract_attributes_along_line(args.map_name, output_shp_path, \
                                                      patch_path=args.cropped_image_dir, roi_buffer=30)
#                 output_shp_attr_path = output_shp_path[:-4] + '_attr.shp'
#                 write_shp_in_imgcoord_with_attr(output_shp_attr_path, line_dict, legend_text=description, image_coords=True)

        if 'thrust' in description.lower() or 'thrust' in sym_name.lower():            
            args.checkpoint = f'{args.trained_model_dir}/thrust_fault_line_model.pt'
            args.line_feature_name = 'thrust_fault_line'
            if args.predict_raster:
                predict_png(args)
            if args.predict_vector:
                predict_shp(args)
#                 line_dict = extract_attributes_along_line(args.map_name, output_shp_path, \
                                                      patch_path=args.cropped_image_dir, roi_buffer=30)
#                 output_shp_attr_path = output_shp_path[:-4] + '_attr.shp'
#                 write_shp_in_imgcoord_with_attr(output_shp_attr_path, line_dict, legend_text=description, image_coords=True)

