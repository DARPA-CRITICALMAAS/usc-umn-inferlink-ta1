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
parser.add_argument('--test_dir', type=str, default='/data/weiweidu/criticalmaas_data/validation_fault_line_comb',
                       help='the path to the folder storing the test maps')
parser.add_argument('--map_name', type=str, default=None)
parser.add_argument('--buffer', type=int, default=10,
                        help='the buffer size for nodes conflation')

if __name__ == '__main__':
    args = parser.parse_args()
    selected_maps = ['NV_HiddenHills'] # 'AZ_GrandCanyon', 'CA_NV_DeathValley'
#     selected_maps = ['AK_Dillingham'] #'AK_Dillingham', 'CA_Elsinore', 'AR_StJoe', 'AK_Kechumstuk'
#     selected_maps = ['CA_Elsinore'] # CO_Alamosa (normal fl)
#     selected_maps = [ 'AK_Hughes', 'OR_Camas'] #, 'AK_Hughes', 'AZ_PipeSpring'
#     selected_maps = ['NV_HiddenHills', 'NM_Sunshine', 'AR_Maumee', 'NM_Volcanoes']
#     selected_maps = ['OR_Carlton', 'NV_HiddenHills', 'NM_Volcanoes', 'CO_Alamosa', 'AK_Kechumstuk']
#     selected_maps = ['AZ_PipeSpring','OR_Camas', 'AZ_PrescottNF', 'MN', 'AR_Maumee']
    cnt = 0
    for root, dirs, files in os.walk(args.test_dir, topdown=False):
        for map_name in files:
            if 'fault' in map_name:# or 'MN' in map_name:
                continue
            if selected_maps != [] and map_name[:-4] not in selected_maps:
                continue
            cnt += 1
            if cnt > 100:
                break
            args.map_name = map_name
            print('----- predicting {} -----'.format(map_name[:-4]))
#             predict_png(args)
            predict_shp(args)