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


def polygon_schema_worker(this_abbr, info_for_this_poly, linking_ids, candidate_info, map_name, fname, dir_to_raster_polygon, dir_to_integrated_output, targeted_crs):
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

    get_referenced_text = False

    chronology_age = np.array(chronology_age)
    chronology_epoch = np.array(chronology_epoch)
    chronology_period = np.array(chronology_period)
    chronology_age_b_int = np.array(chronology_age_b_int)
    chronology_age_t_int = np.array(chronology_age_t_int)

    b_epoch = chronology_period.shape[0]
    t_epoch = -1
    b_interval = ''
    t_interval = ''
    b_age = None
    t_age = None

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
            b_interval = chronology_age[b_epoch]
            t_interval = chronology_age[t_epoch]
            b_age = int(chronology_age_b_int[b_epoch])
            t_age = int(chronology_age_t_int[t_epoch])
    
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
                                                    ], crs=targeted_crs)
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
                mirrored_polygon['GeologicUnit'] = [{'name':None, 'description':None, 'comments':None, 'age_text':str(b_interval)+' - '+str(t_interval), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':int(t_age), 'b_age':int(b_age), 'lithology':None} for _ in range(mirrored_polygon.shape[0])]
            else:
                mirrored_polygon['PolygonType'] = [{'id':int(info_for_this_poly['id'].values[0]), 'name':str(info_for_this_poly['name'].values[0]), 'color':str(info_for_this_poly['color'].values[0]), 'pattern':str(info_for_this_poly['pattern'].values[0]), 'abbreviation':str(info_for_this_poly['abbreviation'].values[0]), 'description':str(info_for_this_poly['description'].values[0]), 'category':str(info_for_this_poly['category'].values[0])} for _ in range(mirrored_polygon.shape[0])]
                mirrored_polygon['GeologicUnit'] = [{'name':None, 'description':None, 'comments':None, 'age_text':str(b_interval)+' - '+str(t_interval), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':int(t_age), 'b_age':int(b_age), 'lithology':None} for _ in range(mirrored_polygon.shape[0])]
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

    # Simplify the vectorization results
    mirrored_polygon = mirrored_polygon[mirrored_polygon['geometry'].area/ 10**6 > 0.001]
    mirrored_polygon['id'] = range(0, mirrored_polygon.shape[0])

    for index, poi in mirrored_polygon.iterrows():
        this_reduced_polygon = poi['geometry'].simplify(0.9, preserve_topology=True)
        this_reduced_polygon = this_reduced_polygon.buffer(2).buffer(-4).buffer(2)
        this_reduced_polygon = this_reduced_polygon.simplify(0.9, preserve_topology=True)
        this_reduced_polygon = shapely.wkt.loads(shapely.wkt.dumps(this_reduced_polygon, trim=True, rounding_precision=0))
        mirrored_polygon.loc[index, 'geometry'] = this_reduced_polygon
    
    mirrored_polygon = mirrored_polygon.explode()
    mirrored_polygon = mirrored_polygon[mirrored_polygon['geometry'].area/ 10**6 > 0.001]
    mirrored_polygon['id'] = range(0, mirrored_polygon.shape[0])
    
    #mirrored_polygon = mirrored_polygon.drop('level_0', axis=1)
    #mirrored_polygon = mirrored_polygon.drop('level_1', axis=1)
    mirrored_polygon.reset_index(inplace=True)
    mirrored_polygon = mirrored_polygon.drop('level_0', axis=1)
    mirrored_polygon = mirrored_polygon.drop('level_1', axis=1)
    #print(mirrored_polygon.keys()) 

    mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
    mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, map_name, fname.replace('_predict.png', '_PolygonFeature.geojson')), driver='GeoJSON')
    #mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, map_name, 'poly_feature_'+str(int(this_abbr))+'.geojson'), driver='GeoJSON')

    return True