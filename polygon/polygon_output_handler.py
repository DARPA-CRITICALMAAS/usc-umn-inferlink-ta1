import numpy as np
import os
import cv2

from geopandas import GeoDataFrame
from shapely import geometry
from shapely.geometry import Polygon
import shapely.wkt

import csv

import geopandas as gpd
from osgeo import ogr, gdal, osr
import pandas as pd

import warnings
warnings.filterwarnings("ignore")




def polygon_output_handler():
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

    polygon_type_db = gpd.read_file(path_to_legend_solution, driver='GeoJSON')


    polygon_feature_counter = 0    
    for fname in os.listdir(dir_to_raster_polygon):    # change directory as needed
        if os.path.isfile(os.path.join(dir_to_raster_polygon, fname)):
            #print(os.path.join(dir_to_raster_polygon, fname), map_name.replace('.tif', '_'))
            if '_predict.png' in fname and map_name.replace('.tif', '_') in fname:
                this_abbr = fname.split('_')[-3]
                #print(this_abbr)
                info_for_this_poly = polygon_type_db[(polygon_type_db['abbreviation'] == this_abbr)]
                #print(info_for_this_poly)
                #print(info_for_this_poly.shape[0])

                b_epoch = chronology_period.shape[0]
                t_epoch = -1
                b_interval = ''
                t_interval = ''
                b_age = ''
                t_age = ''

                if info_for_this_poly.shape[0] > 0:
                    if info_for_this_poly['name'].values.shape[0] > 0 and info_for_this_poly['description'].values.shape[0] > 0:
                        testing_string = str(info_for_this_poly['name'].values[0]) + ': ' + str(info_for_this_poly['description'].values[0])
                    elif info_for_this_poly['name'].values.shape[0] > 0:
                        testing_string = str(info_for_this_poly['name'].values[0])
                    else:
                        testing_string = ''
                    
                    epoch_check = np.flatnonzero(np.core.defchararray.find(testing_string, chronology_period)!=-1)
                    if epoch_check.shape[0] > 0:
                        b_epoch = max(epoch_check)
                        t_epoch = min(epoch_check)
                    
                    epoch_check = np.flatnonzero(np.core.defchararray.find(testing_string, chronology_epoch)!=-1)
                    if epoch_check.shape[0] > 0:
                        b_epoch = min(b_epoch ,max(epoch_check))
                        t_epoch = max(t_epoch, min(epoch_check))

                    epoch_check = np.flatnonzero(np.core.defchararray.find(testing_string, chronology_age)!=-1)
                    if epoch_check.shape[0] > 0:
                        b_epoch = min(b_epoch ,max(epoch_check))
                        t_epoch = max(t_epoch, min(epoch_check))

                    #print(testing_string,b_epoch, t_epoch, b_interval, t_interval, b_age, t_age)
                    if b_epoch != chronology_period.shape[0] and t_epoch != -1:
                        b_interval = chronology_epoch[b_epoch]
                        t_interval = chronology_epoch[t_epoch]
                        b_age = chronology_age[b_epoch]
                        t_age = chronology_age[t_epoch]
                
                in_path = os.path.join(dir_to_raster_polygon, fname)
                base_image = cv2.imread(in_path)
                base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

                out_path = os.path.join(dir_to_integrated_output, map_name, fname.replace('_predict.png', '.geojson'))

                src_ds = gdal.Open(in_path)
                srcband = src_ds.GetRasterBand(1)
                dst_layername = 'polygon'
                drv = ogr.GetDriverByName('geojson')
                dst_ds = drv.CreateDataSource(out_path)

                sp_ref = osr.SpatialReference()
                sp_ref.SetFromUserInput('EPSG:3857')

                dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )
                gdal.Polygonize( srcband, None, dst_layer, 0, [], callback=None )

                del src_ds
                del dst_ds


                
                mirrored_polygon = gpd.GeoDataFrame(columns=['id', 'name', 'geometry', 
                                                                'PolygonType', #: {'id', 'name', 'color', 'pattern', 'abbreviation', 'description', 'category'},  
                                                                'GeologicUnit'#: {'name', 'description', 'comments', 'age_text', 't_interval', 'b_interval', 't_age', 'b_age', 'lithology'}
                                                                ], crs=polygon_type_db.crs)
                polygon_extraction = gpd.read_file(os.path.join(dir_to_integrated_output, map_name, fname.replace('_predict.png', '.geojson')), driver='GeoJSON')
                

                for index, poi in polygon_extraction.iterrows():
                    if index == polygon_extraction.shape[0]-1:
                        break
                    this_mirrored_polygon = shapely.wkt.loads(str(poi['geometry']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))

                    if info_for_this_poly.shape[0] != 1:
                        updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':this_mirrored_polygon, 
                                                                'PolygonType': {'id':None, 'name':None, 'color':None, 'pattern':None, 'abbreviation':None, 'description':None, 'category':None},  
                                                                'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None}}])
                    else:
                        if b_epoch != chronology_period.shape[0]:
                            updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':this_mirrored_polygon, 
                                                                'PolygonType': {'id':int(info_for_this_poly['id'].values[0]), 'name':str(info_for_this_poly['name'].values[0]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(info_for_this_poly['abbreviation'].values[0]), 'description':str(info_for_this_poly['description'].values[0]), 'category':str(info_for_this_poly['category'].values[0])}, 
                                                                'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':str(b_age)+' - '+str(t_age), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':str(t_age), 'b_age':str(b_age), 'lithology':None}}])
                        else:
                            updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':this_mirrored_polygon, 
                                                                'PolygonType': {'id':int(info_for_this_poly['id'].values[0]), 'name':str(info_for_this_poly['name'].values[0]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(info_for_this_poly['abbreviation'].values[0]), 'description':str(info_for_this_poly['description'].values[0]), 'category':str(info_for_this_poly['category'].values[0])}, 
                                                                'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None}}])

                    mirrored_polygon = gpd.GeoDataFrame(pd.concat( [mirrored_polygon, updated_record], ignore_index=True), crs=polygon_type_db.crs)

                    polygon_feature_counter += 1
                
                #print(mirrored_polygon)

                mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
                mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, fname.replace('_predict.png', '_PolygonFeature.geojson')), driver='GeoJSON')






path_to_source = 'Data/OR_Camas.tif' # raster tif
path_to_legend_solution = 'Segmentation_Output/OR_Carlton/OR_Carlton_PolygonType.geojson' # geojson with properties => suffix: _PolygonType.geojson
path_to_groundtruth_legend = 'Data/OR_Camas.json' # json listing all map keys => will be the same as the previous file

dir_to_raster_polygon = 'LOAM_Intermediate/predict/cma/'
dir_to_integrated_output = 'Vectorization_Output/OR_Camas'

target_map_name = 'OR_Camas'


def output_handler(input_path_to_tif, input_path_to_legend_solution, input_path_to_groundtruth_legend, input_dir_to_raster_polygon, input_dir_to_integrated_output):
    global path_to_source
    global path_to_legend_solution
    global path_to_groundtruth_legend
    global dir_to_raster_polygon
    global dir_to_integrated_output
    global target_map_name

    path_to_source = input_path_to_tif
    path_to_legend_solution = input_path_to_legend_solution
    path_to_groundtruth_legend = input_path_to_groundtruth_legend
    dir_to_raster_polygon = input_dir_to_raster_polygon
    dir_to_integrated_output = input_dir_to_integrated_output

    path_list = path_to_source.replace('\\','/').split('/')
    target_map_name = os.path.splitext(path_list[-1])[0]

    polygon_output_handler()
    print('Vectorized outputs are settled at... ', dir_to_integrated_output)




def main():
    global path_to_source
    global path_to_legend_solution
    global path_to_groundtruth_legend
    global dir_to_raster_polygon
    global dir_to_integrated_output
    global target_map_name

    path_to_source = args.path_to_source
    path_to_legend_solution = args.path_to_legend_solution
    path_to_groundtruth_legend = args.path_to_groundtruth_legend
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
    parser.add_argument('--path_to_groundtruth_legend', type=str, default='Data/OR_Camas.json')
    parser.add_argument('--dir_to_raster_polygon', type=str, default='LOAM_Intermediate/predict/cma/')
    parser.add_argument('--dir_to_integrated_output', type=str, default='Vectorization_Output/OR_Camas')
    #parser.add_argument('--targeted_map_list', type=str, default='targeted_map.csv')

    args = parser.parse_args()
    
    main()


