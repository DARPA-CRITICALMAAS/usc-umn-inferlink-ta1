import os
import json
import subprocess

def get_gcp_from_geojson(input_path):
    with open(input_path, 'r') as input_file:
        gcp_dict = json.load(input_file)
    geo_gcp_list = [gcp_dict['geo']['left'], gcp_dict['geo']['right'], gcp_dict['geo']['top'], gcp_dict['geo']['bottom']]
    img_gcp_list = [gcp_dict['img']['img_left'], gcp_dict['img']['img_right'], \
                    gcp_dict['img']['img_top'], gcp_dict['img']['img_bottom']]
    return [img_gcp_list, geo_gcp_list]

def run_gdal_translate(gcp, input_tif_path, output_temp_tif_path):
    geo_left, geo_right, geo_top, geo_bottom = gcp[1]
    img_left, img_right, img_top, img_bottom = gcp[0]
    
    gcp_pair0 = [img_left, img_top, geo_left, geo_top]
    gcp_pair1 = [img_right, img_top, geo_right, geo_top]
    gcp_pair2 = [img_left, img_bottom, geo_left, geo_bottom]
    gcp_pair3 = [img_right, img_bottom, geo_right, geo_bottom]
    
    command = f"gdal_translate -of GTiff -gcp {gcp_pair0[0]} {gcp_pair0[1]} {gcp_pair0[2]} {gcp_pair0[3]} -gcp {gcp_pair1[0]} {gcp_pair1[1]} {gcp_pair1[2]} {gcp_pair1[3]} -gcp {gcp_pair2[0]} {gcp_pair2[1]} {gcp_pair2[2]} {gcp_pair3[3]} -gcp {gcp_pair3[0]} {gcp_pair3[1]} {gcp_pair3[2]} {gcp_pair3[3]} \'{input_tif_path}\' \'{output_temp_tif_path}\'"
    
    res = subprocess.run(command, capture_output=True, text=True, shell=True)
    print(res)
    return res

def run_gdalwrap(input_temp_tif_path, output_tif_path):
    command = f"gdalwarp -r near -t_srs EPSG:4326 -of GTiff \'{input_temp_tif_path}\' \'{output_tif_path}\'"
    res = subprocess.run(command, capture_output=True, text=True, shell=True)
    print(res)
    return res

def run_georeference_map(map_name, input_map_dir, gcp_path, geo_tif_output_dir, temp_dir='./temp'):
    print(f'*** processing {map_name} ***')

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    if not os.path.exists(gcp_path):
        return
    input_tif_path = f'{input_map_dir}/{map_name}.tif'
    interm_tif_path = f'{temp_dir}/{map_name}_temp.tif'
    output_tif_path = f'{geo_tif_output_dir}/{map_name}.tif'

    gcp_pairs = get_gcp_from_geojson(gcp_path)

    run_gdal_translate(gcp_pairs, input_tif_path, interm_tif_path)

    run_gdalwrap(interm_tif_path, output_tif_path)
        
if __name__ == '__main__':
    input_dir = '/data/weiweidu/criticalmaas_data/hackathon2/update_nickel_maps'
    gcp_dir = '/data/weiweidu/usc-umn-inferlink-ta1_local/georeferencing/text-based/output_georef'
#     gcp_dir = '/data/weiweidu/criticalmaas_data/hackathon2/update_nickel_maps_georef_outputs'
    nogeo_tif_dir = '/data/weiweidu/criticalmaas_data/hackathon2/update_nickel_maps'
    temp_dir = '/data/weiweidu/usc-umn-inferlink-ta1_local/system/gpkg/temp'
    
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    geo_tif_output_dir = '/data/weiweidu/criticalmaas_data/hackathon2/update_nickel_maps_georef_maps'
    
    for f_name in os.listdir():
        map_name = f_name[:-4]
        run_georeference_map(map_name, input_dir, gcp_dir, temp_dir, geo_tif_output_dir)

    