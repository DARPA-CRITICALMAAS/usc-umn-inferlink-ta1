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
    if not os.path.exists(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate')):
        os.makedirs(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate'))
    if not os.path.exists(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name)):
        os.makedirs(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name))
    

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
    for fname in os.listdir(dir_to_raster_polygon):    # change directory as needed
        if os.path.isfile(os.path.join(dir_to_raster_polygon, fname)):
            #print(os.path.join(dir_to_raster_polygon, fname), map_name.replace('.tif', '_'))
            if '_predict.png' in fname and map_name.replace('.tif', '_') in fname:
                this_abbr = fname.split('_')[-3]
                #print(this_abbr)
                info_for_this_poly = polygon_type_db[(polygon_type_db['id'] == this_abbr)]
                #print(info_for_this_poly)
                #print(info_for_this_poly.shape[0])

                get_referenced_text = False

                b_epoch = chronology_period.shape[0]
                t_epoch = -1
                b_interval = ''
                t_interval = ''
                b_age = ''
                t_age = ''

                if info_for_this_poly.shape[0] > 0:
                    if int(this_abbr) < linking_ids.shape[0] and linking_ids[int(this_abbr)] != -1:
                        get_referenced_text = True
                        testing_string = candidate_info[linking_ids[int(this_abbr)]][0]
                    else:
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

                out_path = os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '.geojson'))

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
                polygon_extraction = gpd.read_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '.geojson')), driver='GeoJSON')
                

                mirrored_polygon = polygon_extraction.copy()
                mirrored_polygon['id'] = range(0, mirrored_polygon.shape[0])
                mirrored_polygon.drop(mirrored_polygon[mirrored_polygon['id'] == (mirrored_polygon.shape[0]-1)].index, inplace = True)
                mirrored_polygon['name'] = 'PolygonFeature'
                if info_for_this_poly.shape[0] != 1:
                    mirrored_polygon['PolygonType'] = [{'id':None, 'name':None, 'color':None, 'pattern':None, 'abbreviation':None, 'description':None, 'category':None} for _ in range(mirrored_polygon.shape[0])]
                    mirrored_polygon['GeologicUnit'] = [{'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None} for _ in range(mirrored_polygon.shape[0])]
                else:
                    if b_epoch != chronology_period.shape[0]:
                        if get_referenced_text == True:
                            mirrored_polygon['PolygonType'] = [{'id':int(info_for_this_poly['id'].values[0]), 'name':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'description':str(candidate_info[linking_ids[int(this_abbr)]][0]), 'category':str(info_for_this_poly['category'].values[0])} for _ in range(mirrored_polygon.shape[0])]
                            mirrored_polygon['GeologicUnit'] = [{'name':None, 'description':None, 'comments':None, 'age_text':str(b_age)+' - '+str(t_age), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':str(t_age), 'b_age':str(b_age), 'lithology':None} for _ in range(mirrored_polygon.shape[0])]
                        else:
                            mirrored_polygon['PolygonType'] = [{'id':int(info_for_this_poly['id'].values[0]), 'name':str(info_for_this_poly['name'].values[0]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(info_for_this_poly['abbreviation'].values[0]), 'description':str(info_for_this_poly['description'].values[0]), 'category':str(info_for_this_poly['category'].values[0])} for _ in range(mirrored_polygon.shape[0])]
                            mirrored_polygon['GeologicUnit'] = [{'name':None, 'description':None, 'comments':None, 'age_text':str(b_age)+' - '+str(t_age), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':str(t_age), 'b_age':str(b_age), 'lithology':None} for _ in range(mirrored_polygon.shape[0])]
                    else:
                        if get_referenced_text == True:
                            mirrored_polygon['PolygonType'] = [{'id':int(info_for_this_poly['id'].values[0]), 'name':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'description':str(candidate_info[linking_ids[int(this_abbr)]][0]), 'category':str(info_for_this_poly['category'].values[0])} for _ in range(mirrored_polygon.shape[0])]
                            mirrored_polygon['GeologicUnit'] = [{'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None} for _ in range(mirrored_polygon.shape[0])]
                        else:
                            mirrored_polygon['PolygonType'] = [{'id':int(info_for_this_poly['id'].values[0]), 'name':str(info_for_this_poly['name'].values[0]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(info_for_this_poly['abbreviation'].values[0]), 'description':str(info_for_this_poly['description'].values[0]), 'category':str(info_for_this_poly['category'].values[0])} for _ in range(mirrored_polygon.shape[0])]
                            mirrored_polygon['GeologicUnit'] = [{'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None} for _ in range(mirrored_polygon.shape[0])]

                '''
                for index, poi in polygon_extraction.iterrows():
                    if index == polygon_extraction.shape[0]-1:
                        break
                    #this_mirrored_polygon = shapely.wkt.loads(str(poi['geometry']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
                    #this_mirrored_polygon = shapely.wkt.loads(str(poi['geometry']))

                    if info_for_this_poly.shape[0] != 1:
                        updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':poi['geometry'], 
                                                                'PolygonType': {'id':None, 'name':None, 'color':None, 'pattern':None, 'abbreviation':None, 'description':None, 'category':None},  
                                                                'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None}}])
                    else:
                        if b_epoch != chronology_period.shape[0]:
                            if get_referenced_text == True:
                                updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':poi['geometry'], 
                                                                    'PolygonType': {'id':int(info_for_this_poly['id'].values[0]), 'name':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'description':str(candidate_info[linking_ids[int(this_abbr)]][0]), 'category':str(info_for_this_poly['category'].values[0])}, 
                                                                    'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':str(b_age)+' - '+str(t_age), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':str(t_age), 'b_age':str(b_age), 'lithology':None}}])
                            else:
                                updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':poi['geometry'], 
                                                                    'PolygonType': {'id':int(info_for_this_poly['id'].values[0]), 'name':str(info_for_this_poly['name'].values[0]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(info_for_this_poly['abbreviation'].values[0]), 'description':str(info_for_this_poly['description'].values[0]), 'category':str(info_for_this_poly['category'].values[0])}, 
                                                                    'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':str(b_age)+' - '+str(t_age), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':str(t_age), 'b_age':str(b_age), 'lithology':None}}])
                        else:
                            if get_referenced_text == True:
                                updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':poi['geometry'], 
                                                                    'PolygonType': {'id':int(info_for_this_poly['id'].values[0]), 'name':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(candidate_info[linking_ids[int(this_abbr)]][1]), 'description':str(candidate_info[linking_ids[int(this_abbr)]][0]), 'category':str(info_for_this_poly['category'].values[0])}, 
                                                                    'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None}}])
                            else:
                                updated_record = gpd.GeoDataFrame([{'id':polygon_feature_counter, 'name':'PolygonFeature', 'geometry':poi['geometry'], 
                                                                    'PolygonType': {'id':int(info_for_this_poly['id'].values[0]), 'name':str(info_for_this_poly['name'].values[0]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(info_for_this_poly['abbreviation'].values[0]), 'description':str(info_for_this_poly['description'].values[0]), 'category':str(info_for_this_poly['category'].values[0])}, 
                                                                    'GeologicUnit': {'name':None, 'description':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None}}])

                    mirrored_polygon = gpd.GeoDataFrame(pd.concat( [mirrored_polygon, updated_record], ignore_index=True), crs=polygon_type_db.crs)

                    polygon_feature_counter += 1
                '''
                #print(mirrored_polygon)
                

                mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
                mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, map_name, fname.replace('_predict.png', '_PolygonFeature.geojson')), driver='GeoJSON')
                #mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, map_name, 'poly_feature_'+str(int(this_abbr))+'.geojson'), driver='GeoJSON')






path_to_source = 'Data/OR_Camas.tif' # raster tif
path_to_legend_solution = 'Segmentation_Output/OR_Carlton/OR_Carlton_PolygonType.geojson' # geojson with properties => suffix: _PolygonType.geojson
path_to_legend_description = 'Segmentation_Output/OR_Carlton/OR_Carlton_polygon.json' # json with text-based information => suffix: _polygon.json
path_to_json = 'Data/OR_Camas.json' # json listing all map keys => will be the same as the previous file

dir_to_raster_polygon = 'LOAM_Intermediate/predict/cma/'
dir_to_integrated_output = 'Vectorization_Output'

target_map_name = 'OR_Camas'


def output_handler(input_path_to_tif, input_path_to_legend_solution, input_path_to_legend_description, input_path_to_json, input_dir_to_raster_polygon, input_dir_to_integrated_output):
    global path_to_source
    global path_to_legend_solution
    global path_to_legend_description
    global path_to_json
    global dir_to_raster_polygon
    global dir_to_integrated_output
    global target_map_name

    path_to_source = input_path_to_tif
    path_to_legend_solution = input_path_to_legend_solution
    path_to_legend_description = input_path_to_legend_description
    path_to_json = input_path_to_json
    dir_to_raster_polygon = input_dir_to_raster_polygon
    dir_to_integrated_output = input_dir_to_integrated_output

    path_list = path_to_source.replace('\\','/').split('/')
    target_map_name = os.path.splitext(path_list[-1])[0]

    polygon_output_handler()
    print('Vectorized outputs are settled at... ', dir_to_integrated_output)




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


