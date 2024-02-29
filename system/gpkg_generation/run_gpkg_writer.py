import os
from gpkg_writer_geo_coord import write_gpkg as write_geo_gpkg
from gpkg_writer_img_coord import write_gpkg as write_img_gpkg
from georeference_map import run_georeference_map
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--output_dir', \
                    default='/data/weiweidu/temp', type=str)
parser.add_argument('--map_name', default='2188_1086', type=str)
parser.add_argument('--layout_output_dir', \
                    default='/data/weiweidu/criticalmaas_data/hackathon2/MN_MT_raw_maps_legend_item_description_outputs', type=str)
parser.add_argument('--georef_output_dir', \
                    default='/data/weiweidu/criticalmaas_data/georeference_output_sample', type=str)
parser.add_argument('--poly_output_dir', \
                    default='/data/weiweidu/criticalmaas_data/hackathon2/nickel_polygon_extraction_demo', type=str)
parser.add_argument('--ln_output_dir', \
                    default='/data/weiweidu/criticalmaas_data/hackathon2/nickel_line_extraction_demo', type=str)
parser.add_argument('--pt_output_dir', \
                    default='/data/weiweidu/criticalmaas_data/hackathon2/nickel_point_extraction_demo', type=str)
parser.add_argument('--nongeoref_map_dir', \
                    default='/data/weiweidu/criticalmaas_data/hackathon2/MN_MT_raw_maps', type=str)
parser.add_argument('--georef_map_output', \
                    default='/data/weiweidu/criticalmaas_data/hackathon2/update_nickel_maps_georef_maps', type=str)
    
if __name__ == '__main__':    
    args = parser.parse_args()
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
     
    map_name = args.map_name
    layout_output_path = os.path.join(args.layout_output_dir, map_name+'_line.json')
    georef_output_path = os.path.join(args.georef_output_dir, map_name+'.json')
    poly_output_dir = os.path.join(args.poly_output_dir, map_name)
    ln_output_dir = os.path.join(args.ln_output_dir, map_name)

    pt_output_path = os.path.join(args.pt_output_dir, map_name+'.geojson')
    nongeoref_map_dir = args.nongeoref_map_dir
    georef_map_output_dir = args.georef_map_output

    if os.path.exists(georef_output_path):
        geotif_path = run_georeference_map(map_name, nongeoref_map_dir, georef_output_path, georef_map_output_dir)
        print(f'results saved in {geotif_path}')
        write_geo_gpkg(output_dir, map_name, geotif_path, layout_output_path, georef_output_path, \
                   poly_output_dir, ln_output_dir, pt_output_path)

#     output_img_coord_dir = output_dir+'_img'
#     if not os.path.exists(output_img_coord_dir):
#         os.mkdir(output_img_coord_dir)

#     write_img_gpkg(output_img_coord_dir, map_name, layout_output_path, georef_output_path, \
#                poly_output_dir, ln_output_dir, pt_output_path)
        