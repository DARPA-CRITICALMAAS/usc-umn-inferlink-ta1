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
    chronology_age_b_int = [0.0042, 0.0082, 0.0117, 0.129, 0.774, 1.80, 2.58, 
                    3.60, 5.333, 7.246, 11.63, 13.82, 15.97, 20.44, 23.03, 
                    27.82, 33.9, 37.71, 41.2, 47.8, 56.0, 59.2, 61.6, 66.0, 
                    72.1, 83.6, 86.3, 89.8, 93.9, 100.5, 113.0, 121.4, 125.77, 132.6, 139.8, 145.0, 
                    149.2, 154.8, 161.5, 165.3, 168.2, 170.9, 174.7, 184.2, 192.9, 199.5, 201.4, 
                    208.5, 227.0, 237.0, 242.0, 247.2, 251.2, 251.9, 
                    254.14, 259.51, 264.28, 266.9, 273.01, 283.5, 290.1, 293.52, 298.9, 
                    303.7, 307.0, 315.2, 323.2, 330.9, 346.7, 358.9, 
                    371.1, 382.7, 387.7, 393.3, 407.6, 410.8, 419.2, 
                    423.0, 425.6, 427.4, 430.5, 433.4, 438.5, 440.8, 443.8, 
                    445.2, 453.0, 458.4, 467.3, 470.0, 477.7, 485.4, 
                    489.5, 494.0, 497.0, 500.5, 504.5, 509.0, 514.0, 521.0, 529.0, 538.8, 
                    635.0, 720.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2050.0, 2300.0, 2500.0, 
                    2800.0, 3200.0, 3600.0, 4031.0, 4567.3
                    ]
    chronology_age_t_int = [0, 0.0042, 0.0082, 0.0117, 0.129, 0.774, 1.80, 
                    2.58, 3.60, 5.333, 7.246, 11.63, 13.82, 15.97, 20.44, 
                    23.03, 27.82, 33.9, 37.71, 41.2, 47.8, 56.0, 59.2, 61.6, 
                    66.0, 72.1, 83.6, 86.3, 89.8, 93.9, 100.5, 113.0, 121.4, 125.77, 132.6, 139.8, 
                    145.0,149.2, 154.8, 161.5, 165.3, 168.2, 170.9, 174.7, 184.2, 192.9, 199.5, 
                    201.4, 208.5, 227.0, 237.0, 242.0, 247.2, 251.2, 
                    251.9, 254.14, 259.51, 264.28, 266.9, 273.01, 283.5, 290.1, 293.52, 
                    298.9, 303.7, 307.0, 315.2, 323.2, 330.9, 346.7, 
                    358.9, 372.2,382.7, 387.7, 393.3, 407.6, 410.8, 
                    419.2, 423.0, 425.6, 427.4, 430.5, 433.4, 438.5, 440.8, 
                    443.8, 445.2, 453.0, 458.4, 467.3, 470.0, 477.7, 
                    485.4, 489.5, 494.0, 497.0, 500.5, 504.5, 509.0, 514.0, 521.0, 529.0, 
                    538.8, 635.0, 720.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2050.0, 2300.0, 
                    2500.0, 2800.0, 3200.0, 3600.0, 4031.0
                    ]
    chronology_age = ['Meghalayan', 'Northgrippian', 'Greenlandian', 'Late Pleistocene', 'Chibanian', 'Calabrian', 'Gelasian', 
                    'Piacenzian', 'Zanclean', 'Messinian', 'Tortonian', 'Serravallian', 'Langhian', 'Burdigalian', 'Aquitanian', 
                    'Chattian', 'Rupelian', 'Priabonian', 'Bartonian', 'Lutetian', 'Ypresian', 'Thanetian', 'Selandian', 'Danian', 
                    'Maastrichtian', 'Campanian', 'Santonian', 'Coniacian', 'Turonian', 'Cenomanian', 'Albian', 'Aptian', 'Barremian', 'Hauterivian', 'Valanginian', 'Berriasian', 
                    'Tithonian', 'Kimmeridgian', 'Oxfordian', 'Callovian', 'Bathonian', 'Bajocian', 'Aalenian', 'Toarcian', 'Pliensbachian', 'Sinemurian', 'Hettangian', 
                    'Rhaetian', 'Norian', 'Carnian', 'Ladinian', 'Anisian', 'Olenekian', 'Induan', 
                    'Changhsingian', 'Wuchiapingian', 'Capitanian', 'Wordian', 'Roadian', 'Kungurian', 'Artinskian', 'Sakmarian', 'Asselian', 
                    'Gzhelian', 'Kasimovian', 'Moscovian', 'Bashkirian', 'Serpukhovian', 'Viséan', 'Tournaisian', 
                    'Famennian', 'Frasnian', 'Givetian', 'Eifelian', 'Emsian', 'Pragian', 'Lochkovian', 
                    'Pridoli', 'Ludfordian', 'Gorstian', 'Homerian', 'Sheinwoodian', 'Telychian', 'Aeronian', 'Rhuddanian', 
                    'Hirnantian', 'Katian', 'Sandbian', 'Darriwilian', 'Dapingian', 'Floian', 'Tremadocian', 
                    'Stage 10', 'Jiangshanian', 'Paibian', 'Guzhangian', 'Drumian', 'Wuliuan', 'Stage 4', 'Stage 3', 'Stage 2', 'Fortunian', 
                    'Ediacaran', 'Cryogenian', 'Tonian', 'Stenian', 'Ectasian', 'Calymmian', 'Statherian', 'Orosirian', 'Rhyacian', 'Siderian', 
                    'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean'
                    ]
    chronology_epoch = ['Holocene', 'Holocene', 'Holocene', 'Pleistocene', 'Pleistocene', 'Pleistocene', 'Pleistocene', 
                    'Pliocene', 'Pliocene', 'Miocene', 'Miocene', 'Miocene', 'Miocene', 'Miocene', 'Miocene', 
                    'Oligocene', 'Oligocene', 'Eocene', 'Eocene', 'Eocene', 'Eocene', 'Paleocene', 'Paleocene', 'Paleocene', 
                    'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Late Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 'Early Cretaceous', 
                    'Late Jurassic', 'Late Jurassic', 'Late Jurassic', 'Middle Jurassic', 'Middle Jurassic', 'Middle Jurassic', 'Middle Jurassic', 'Early Jurassic', 'Early Jurassic', 'Early Jurassic', 'Early Jurassic', 
                    'Late Triassic', 'Late Triassic', 'Late Triassic', 'Middle Triassic', 'Middle Triassic', 'Early Triassic', 'Early Triassic', 
                    'Lopingian', 'Lopingian', 'Guadalupian', 'Guadalupian', 'Guadalupian', 'Cisuralian', 'Cisuralian', 'Cisuralian', 'Cisuralian', 
                    'Pennsylvanian', 'Pennsylvanian', 'Pennsylvanian', 'Pennsylvanian', 'Mississippian', 'Mississippian', 'Mississippian', 
                    'Late Devonian', 'Late Devonian', 'Middle Devonian', 'Middle Devonian', 'Early Devonian', 'Early Devonian', 'Early Devonian', 
                    'Pridoli', 'Ludlow', 'Ludlow', 'Wenlock', 'Wenlock', 'Llandovery', 'Llandovery', 'Llandovery', 
                    'Late Ordovician', 'Late Ordovician', 'Late Ordovician', 'Middle Ordovician', 'Middle Ordovician', 'Early Ordovician', 'Early Ordovician', 
                    'Furongian', 'Furongian', 'Furongian', 'Miaolingian', 'Miaolingian', 'Miaolingian', 'Series 2', 'Series 2', 'Terreneuvian', 'Terreneuvian', 
                    'Ediacaran', 'Cryogenian', 'Tonian', 'Stenian', 'Ectasian', 'Calymmian', 'Statherian', 'Orosirian', 'Rhyacian', 'Siderian', 
                    'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean',
                    ]
    chronology_period = ['Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', 'Quaternary', 
                    'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 'Neogene', 
                    'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 'Paleogene', 
                    'Cretaceous', 'Cretaceous', 'LCretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 'Cretaceous', 
                    'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 'Jurassic', 
                    'Triassic', 'Triassic', 'Triassic', 'Triassic', 'Triassic', 'Triassic', 'Triassic', 
                    'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 'Permian', 
                    'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 'Carboniferous', 
                    'Devonian', 'Devonian', 'Devonian', 'Devonian', 'Devonian', 'Devonian', 'Devonian', 
                    'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 'Silurian', 
                    'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 'Ordovician', 
                    'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 'Cambrian', 
                    'Ediacaran', 'Cryogenian', 'Tonian', 'Stenian', 'Ectasian', 'Calymmian', 'Statherian', 'Orosirian', 'Rhyacian', 'Siderian', 
                    'Neoarchean', 'Mesoarchean', 'Paleoarchean', 'Eoarchean', 'Hadean',
                    ]


    chronology_age = np.array(chronology_age)
    chronology_epoch = np.array(chronology_epoch)
    chronology_period = np.array(chronology_period)

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
    print(len(info_set))



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


def output_handler(input_path_to_tif, input_path_to_legend_solution, input_path_to_legend_description, input_path_to_json, input_dir_to_raster_polygon, input_dir_to_integrated_output, input_dir_to_raster_output, input_vectorization):
    global path_to_source
    global path_to_legend_solution
    global path_to_legend_description
    global path_to_json
    global dir_to_raster_polygon
    global dir_to_integrated_output
    global dir_to_raster_output
    global target_map_name

    path_to_source = input_path_to_tif
    path_to_legend_solution = input_path_to_legend_solution
    path_to_legend_description = input_path_to_legend_description
    path_to_json = input_path_to_json
    dir_to_raster_polygon = input_dir_to_raster_polygon
    dir_to_integrated_output = input_dir_to_integrated_output
    dir_to_raster_output = input_dir_to_raster_output

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


