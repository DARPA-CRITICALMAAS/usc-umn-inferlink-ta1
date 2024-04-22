import numpy as np
import os
import cv2

from geopandas import GeoDataFrame
from shapely import geometry
from shapely.geometry import Polygon
import shapely.wkt

import csv
import json
from scipy import spatial

import geopandas as gpd
from osgeo import ogr, gdal, osr
import pandas as pd

import multiprocessing
import shutil

import warnings
warnings.filterwarnings("ignore")

import postprocessing_workers.geojson_to_tif as geojson_to_tif
import postprocessing_workers.polygon_schema_worker as polygon_schema_worker
PROCESSES = 10


def polygon_output_handler():
    map_name = target_map_name

    if not os.path.exists(dir_to_integrated_output):
        os.makedirs(dir_to_integrated_output)
    if not os.path.exists(os.path.join(dir_to_integrated_output, map_name)):
        os.makedirs(os.path.join(dir_to_integrated_output, map_name))
    if not os.path.exists(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate')):
        os.makedirs(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate'))
    if not os.path.exists(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name)):
        os.makedirs(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name))
    if not os.path.exists(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_0')):
        os.makedirs(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_0'))
    if not os.path.exists(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_1')):
        os.makedirs(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_1'))
    if not os.path.exists(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_2')):
        os.makedirs(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_2'))

    if not os.path.exists(os.path.join(dir_to_raster_output, map_name)):
        os.makedirs(os.path.join(dir_to_raster_output, map_name))
    

    link_description = True
    if os.path.isfile(path_to_legend_solution) == False:
        print('Please provide the output file from legend-item segmentation that has the suffix of "_PolygonType.geojson"...')
        link_description = False
    if os.path.isfile(path_to_legend_description) == False:
        print('Please provide the output file from legend-item description that has the suffix of "_polygon.json" if you intend to do so...')
        link_description = False
    
    # Setup referencing text-descriptions
    linking_ids = []
    candidate_list = []
    candidate_info = []

    if link_description == True:
        with open(path_to_legend_description) as f:
            gj = json.load(f)
            #print(gj)

            for this_key, this_row in gj.items():
                if ',' in this_key:
                    xy_list = this_key[1:-1].split(',')
                    center_x = int((float(xy_list[0]) + float(xy_list[2])) / 2.0)
                    center_y = int((float(xy_list[1]) + float(xy_list[3])) / 2.0)

                    #print(this_key, center_x, center_y, this_row['description'], this_row['symbol name'])
                    candidate_list.append([center_x, center_y])
                    candidate_info.append([this_row['description'], this_row['symbol name']])
        candidate_list = np.array(candidate_list)
        candidate_info = np.array(candidate_info)


        if candidate_list.shape[0] > 0:
            with open(path_to_json) as f:
                gj = json.load(f)
                #print(gj)

                for this_gj in gj['shapes']:
                    #print(this_gj)
                    names = this_gj['label']
                    features = this_gj['points']

                    if '_poly' in names:
                        center_x = int((float(features[0][0]) + float(features[1][0])) / 2.0)
                        center_y = int((float(features[0][1]) + float(features[1][1])) / 2.0)
                        #print(names, center_x, center_y)
                        this_pt = [center_x, center_y]

                        distance,index = spatial.KDTree(candidate_list).query(this_pt)
                        #print(distance, index)

                        #print(names, center_x, center_y, candidate_list[index], distance)
                        if distance < 6.6:
                            #print(candidate_info[index])
                            linking_ids.append(index)
                        else:
                            linking_ids.append(-1)
                        pass
        else:
            link_description = False

    linking_ids = np.array(linking_ids)
    #print(linking_ids)


    if os.path.isfile(path_to_legend_solution) == False:
        polygon_feature_counter = 0
        info_set = []
        for fname in os.listdir(dir_to_raster_polygon):    # change directory as needed
            if os.path.isfile(os.path.join(dir_to_raster_polygon, fname)):
                #print(os.path.join(dir_to_raster_polygon, fname), map_name.replace('.tif', '_'))
                if '_predict.png' in fname and map_name.replace('.tif', '_') in fname:
                    this_abbr = fname.split('_')[-3]
                    #print(this_abbr)
                    info_for_this_poly = np.array([])
                    #print(info_for_this_poly)
                    #print(info_for_this_poly.shape[0])

                    this_info = [this_abbr, info_for_this_poly, fname]
                    #this_info = np.array(this_info)
                    info_set.append(this_info)
    else:
        polygon_type_db = gpd.read_file(path_to_legend_solution, driver='GeoJSON')


        polygon_feature_counter = 0
        info_set = []
        for fname in os.listdir(dir_to_raster_polygon):    # change directory as needed
            if os.path.isfile(os.path.join(dir_to_raster_polygon, fname)):
                #print(os.path.join(dir_to_raster_polygon, fname), map_name.replace('.tif', '_'))
                if '_predict.png' in fname and map_name.replace('.tif', '_') in fname:
                    this_abbr = fname.split('_')[-3]
                    #print(this_abbr)
                    info_for_this_poly = polygon_type_db[(polygon_type_db['id'] == this_abbr)]
                    #print(info_for_this_poly)
                    #print(info_for_this_poly.shape[0])

                    this_info = [this_abbr, info_for_this_poly, fname]
                    #this_info = np.array(this_info)
                    info_set.append(this_info)

        #info_set = np.array(info_set)


    with multiprocessing.Pool(int(PROCESSES)) as pool:
        if os.path.isfile(path_to_legend_solution) == False:
            callback = pool.starmap_async(polygon_schema_worker.polygon_schema_worker, [(info_set[this_poly][0], info_set[this_poly][1], linking_ids, candidate_info, map_name, info_set[this_poly][2], dir_to_raster_polygon, dir_to_integrated_output, None, ) for this_poly in range(0, len(info_set))])
        else:
            callback = pool.starmap_async(polygon_schema_worker.polygon_schema_worker, [(info_set[this_poly][0], info_set[this_poly][1], linking_ids, candidate_info, map_name, info_set[this_poly][2], dir_to_raster_polygon, dir_to_integrated_output, polygon_type_db.crs, ) for this_poly in range(0, len(info_set))])
        multiprocessing_results = callback.get()

        for rec in multiprocessing_results:
            if rec == True:
                polygon_feature_counter = polygon_feature_counter + 1
                if polygon_feature_counter % 10 == 0:
                    print('Finalizing vectorization of polygon features...... ('+str(polygon_feature_counter)+'/'+str(len(info_set))+')')
    


    info_set_v0 = []
    path_to_source_tif = os.path.join(path_to_source)
    example_input = cv2.imread(path_to_source_tif)
    example_input_shape = example_input.shape
    for fname in os.listdir(dir_to_raster_polygon):
        if os.path.isfile(os.path.join(dir_to_raster_polygon, fname)):
            #print(os.path.join(dir_to_raster_polygon, fname), map_name.replace('.tif', '_'))
            if '_predict.png' in fname and map_name.replace('.tif', '_') in fname:
                path_to_extraction_geojson = os.path.join(dir_to_integrated_output, map_name, fname.replace('_predict.png', '_PolygonFeature.geojson'))
                path_to_extraction_tif = os.path.join(dir_to_raster_output, map_name, fname.replace('_predict.png', '_PolygonFeature.tif'))
                info_set_v0.append([path_to_extraction_geojson, path_to_extraction_tif, example_input_shape])
    print(len(info_set_v0))
    print(info_set_v0)


    with multiprocessing.Pool(PROCESSES) as pool:
        #multiprocessing_results = [pool.apply_async(validation_evaluation_worker, (info_id,info_set[info_id],)) for info_id in range(0, len(info_set))]
        callback = pool.starmap_async(geojson_to_tif.geojson_to_tif, [(info_id, info_set_v0[info_id], ) for info_id in range(0, len(info_set_v0))]) # len(info_set)
        multiprocessing_results  = callback.get()
            
        for returned_info in multiprocessing_results:
            #map_name, legend_name, precision, recall, f_score = returned_info
            try:
                returned = returned_info
                get_output_tif = returned[0]
                path_to_output_tif = returned[1]

                if get_output_tif == False:
                    print('geojson not available due to large file size...')
                    print(path_to_output_tif)
            except:
                print('Error occured during geojson_to_tif...')




def copy_raster_output(dir_to_integrated_output):
    map_name = target_map_name
    polygon_feature_counter = 0
    info_set = []
    for fname in os.listdir(dir_to_raster_polygon):    # change directory as needed
        if os.path.isfile(os.path.join(dir_to_raster_polygon, fname)):
            #print(os.path.join(dir_to_raster_polygon, fname), map_name.replace('.tif', '_'))
            if '_predict.png' in fname and map_name.replace('.tif', '_') in fname:
                shutil.copyfile(os.path.join(dir_to_raster_polygon, fname), os.path.join(dir_to_integrated_output, fname))
                



path_to_source = 'Data/OR_Camas.tif' # raster tif
path_to_legend_solution = 'Segmentation_Output/OR_Carlton/OR_Carlton_PolygonType.geojson' # geojson with properties => suffix: _PolygonType.geojson
path_to_legend_description = 'Segmentation_Output/OR_Carlton/OR_Carlton_polygon.json' # json with text-based information => suffix: _polygon.json
path_to_json = 'Data/OR_Camas.json' # json listing all map keys => will be the same as the previous file

dir_to_raster_polygon = 'LOAM_Intermediate/predict/cma/'
dir_to_integrated_output = 'Vectorization_Output'
dir_to_raster_output = 'Raster_Output'

target_map_name = 'OR_Camas'
set_schema = True


def output_handler(input_path_to_tif, input_path_to_legend_solution, input_path_to_legend_description, input_path_to_json, input_dir_to_raster_polygon, input_dir_to_integrated_output, input_dir_to_raster_output, input_vectorization, input_set_schema):
    global path_to_source
    global path_to_legend_solution
    global path_to_legend_description
    global path_to_json
    global dir_to_raster_polygon
    global dir_to_integrated_output
    global dir_to_raster_output
    global target_map_name
    global set_schema

    path_to_source = input_path_to_tif
    path_to_legend_solution = input_path_to_legend_solution
    path_to_legend_description = input_path_to_legend_description
    path_to_json = input_path_to_json
    dir_to_raster_polygon = input_dir_to_raster_polygon
    dir_to_integrated_output = input_dir_to_integrated_output
    dir_to_raster_output = input_dir_to_raster_output
    set_schema = input_set_schema

    path_list = path_to_source.replace('\\','/').split('/')
    target_map_name = os.path.splitext(path_list[-1])[0]

    if input_vectorization == True:
        polygon_output_handler()
        print('Vectorized outputs are settled at... ', dir_to_integrated_output)
    else:
        copy_raster_output(dir_to_integrated_output)




def main():
    global path_to_source
    global path_to_legend_solution
    global path_to_json
    global dir_to_raster_polygon
    global dir_to_integrated_output
    global target_map_name

    path_to_source = args.path_to_source
    path_to_legend_solution = args.path_to_legend_solution
    path_to_json = args.path_to_json
    dir_to_raster_polygon = args.dir_to_raster_polygon
    dir_to_integrated_output = args.dir_to_integrated_output

    path_list = path_to_source.replace('\\','/').split('/')
    target_map_name = os.path.splitext(path_list[-1])[0]

    polygon_output_handler()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_source', type=str, default='Data/OR_Camas.tif')
    parser.add_argument('--path_to_legend_solution', type=str, default='Segmentation_Output/OR_Carlton/OR_Carlton_PolygonType.geojson')
    parser.add_argument('--path_to_json', type=str, default='Data/OR_Camas.json')
    parser.add_argument('--dir_to_raster_polygon', type=str, default='LOAM_Intermediate/predict/cma/')
    parser.add_argument('--dir_to_integrated_output', type=str, default='Vectorization_Output/OR_Camas')
    #parser.add_argument('--targeted_map_list', type=str, default='targeted_map.csv')

    args = parser.parse_args()
    
    main()


