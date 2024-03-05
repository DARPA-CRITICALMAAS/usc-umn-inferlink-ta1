import os, sys
import gpkg_writer_geo_coord 
import gpkg_writer_img_coord
from georeference_map import run_georeference_map
from argparse import ArgumentParser
import logging

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
parser.add_argument('--log_path', type=str, \
                    default='/data/weiweidu/temp/gpkg_writer_logger.log')

args = parser.parse_args()    
logger = logging.getLogger(args.log_path)
handler = logging.FileHandler(f'{args.log_path}', mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
    
    
def main(output_dir, map_name, layout_output_path, georef_output_path, nongeoref_map_dir, \
         poly_output_dir, ln_output_dir, pt_output_dir):    
    
    nongeoref_map_path = os.path.join(nongeoref_map_dir, map_name+'.tif')
    if not os.path.exists(nongeoref_map_path):
        logger.error(f'{nongeoref_map_path} does not exist')
        sys.exit(1)
    
    geo_gpkg_output_dir = os.path.join(output_dir, 'geo_gpkg_output')
    if not os.path.exists(geo_gpkg_output_dir):
        os.mkdir(geo_gpkg_output_dir)
    
    img_gpkg_output_dir = os.path.join(output_dir, 'img_gpkg_output')
    if not os.path.exists(img_gpkg_output_dir):
        os.mkdir(img_gpkg_output_dir)
    
    if os.path.exists(georef_output_path):
        geotif_path = run_georeference_map(map_name, nongeoref_map_dir, georef_output_path, georef_map_output_dir)
        logger.info(f'Georeferenced tif map is saved in {geotif_path}')
        gpkg_writer_geo_coord.write_gpkg(geo_gpkg_output_dir, map_name, layout_output_path, georef_output_path, \
                       poly_output_dir, ln_output_dir, pt_output_dir, logger)
        logger.info(f'Georeferenced GPKG is saved in {geo_gpkg_output_dir}/{map_name}_geo.gpkg')
    else:
        logger.warning(f'No ground control points output from the georeferencing module.')
    
    gpkg_writer_img_coord.write_gpkg(img_gpkg_output_dir, map_name, layout_output_path, \
                       poly_output_dir, ln_output_dir, pt_output_dir, logger)
    logger.info(f'GPKG in the image coordinate is saved in {img_gpkg_output_dir}/{map_name}_img.gpkg')
if __name__ == '__main__':    
    
    output_dir = args.output_dir
    log_path = args.log_path
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
     
    map_name = args.map_name
    layout_output_path = os.path.join(args.layout_output_dir, map_name+'_line.json')
    georef_output_path = os.path.join(args.georef_output_dir, map_name+'.json')
    poly_output_dir = os.path.join(args.poly_output_dir, map_name)
    ln_output_dir = os.path.join(args.ln_output_dir, map_name)
    pt_output_dir = os.path.join(args.pt_output_dir, map_name)
    
    nongeoref_map_dir = args.nongeoref_map_dir
    georef_map_output_dir = args.georef_map_output

    main(output_dir, map_name, layout_output_path, georef_output_path, nongeoref_map_dir, \
         poly_output_dir, ln_output_dir, pt_output_dir)    
