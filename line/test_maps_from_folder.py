from test_darpa_map_conflation_shapely_mask_output_schema import predict_png, predict_shp
from argparse import ArgumentParser
import os
import yaml

parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--checkpoint', default=None, help='checkpoint of the model to test.')
parser.add_argument('--device', default='cuda',
                        help='device to use for training')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[3],
                        help='list of index where skip conn will be made.')
parser.add_argument('--predict_raster', default=False, type=bool, help='predict results are png if True')
parser.add_argument('--predict_vector', default=False, type=bool, help='predict results are geojson if True')
parser.add_argument('--line_feature_name', default='fault_line', type=str, help='the name of line feature')
parser.add_argument('--test_tif_map_dir', type=str, default='/data/weiweidu/criticalmaas_data/validation',
                       help='the path to the folder storing the test tif maps')
parser.add_argument('--test_png_map_dir', type=str, default='/data/weiweidu/criticalmaas_data/validation_fault_line_comb',
                       help='the path to the folder storing the test png maps')
parser.add_argument('--cropped_image_dir', type=str, default='/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines',
                       help='the path to the cropped images')
parser.add_argument('--map_bound_dir', type=str, default='/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines',
                       help='the path to masks for map area')
parser.add_argument('--prediction_dir', type=str, default='./pred_maps',
                       help='the path to save prediction results')
parser.add_argument('--map_name', type=str, default=None)
parser.add_argument('--buffer', type=int, default=10,
                        help='the buffer size for nodes conflation')

def check_path(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")
    except FileNotFoundError as e:
        print(e)

if __name__ == '__main__':
    args = parser.parse_args()
    selected_maps = [] 
    
    check_path(args.test_tif_map_dir)
    check_path(args.test_png_map_dir)
    check_path(args.cropped_image_dir)
    check_path(args.map_bound_dir)
    
    if not os.path.exists(args.prediction_dir):
        os.makedirs(args.prediction_dir)
    cnt = 0
    for root, dirs, files in os.walk(args.test_png_map_dir, topdown=False):
        for map_name in files:
            if args.line_feature_name in map_name:
                continue
            if selected_maps != [] and map_name[:-4] not in selected_maps:
                continue
            cnt += 1
            if cnt > 100:
                break
            args.map_name = map_name
            
            if args.predict_raster:
                print('----- predicting {} raster map -----'.format(map_name[:-4]))
                predict_png(args)
            elif args.predict_vector:
                print('----- predicting {} vector -----'.format(map_name[:-4]))
                predict_shp(args)
            else:
                print('----- predicting {} raster map by default -----'.format(map_name[:-4]))
                predict_png(args)
