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



def rebuilding_json(gj, this_abbr, rounding_off=False):
    # Process to combine 'properties' and 'geometry' at the same level
    combined_features = []
    for feature in gj['features']:
        combined_feature = {**feature['properties'], **feature['geometry']}
        combined_features.append(combined_feature)

    for feature in combined_features:
        polygon_feature = {
            "id": f"polygon_feature_{this_abbr}_instance_{feature['id']}",
            "geometry": {
                # Use the existing 'coordinates' from the feature's geometry
                "coordinates": feature['coordinates']
            },
            "properties": {
                "model": feature.get('model'),  # Using .get() to avoid KeyError if key doesn't exist
                "model_version": feature.get('model_version'),
                "confidence": feature.get('confidence'),
                "model_config": feature.get('model_config'),
            }
        }

        feature['polygon_features'] = {
            "features": [polygon_feature]
        }
    

    for feature in combined_features:
        if 'coordinates' in feature:
            del feature['coordinates']
        if 'model' in feature:
            del feature['model']
        if 'model_version' in feature:
            del feature['model_version']
        if 'confidence' in feature:
            del feature['confidence']
        if 'model_config' in feature:
            del feature['model_config']
        if 'type' in feature:
            del feature['type']
        
        # some formatting fix
        feature['id'] = 'polygon_feature_'+str(this_abbr)+'_instance_'+str(feature['id'])

        if 'legend_bbox' in feature and feature['legend_bbox']:
            bbox_str = feature['legend_bbox'].strip('POLYGON ((').strip('))')
            if rounding_off==True:
                feature['legend_bbox'] = [list(map(int, coord.split())) for coord in bbox_str.split(', ')]
            else:
                bbox_list = [list(map(float, coord.split())) for coord in bbox_str.split(', ')]
                feature['legend_bbox'] = [[round(coord, 1) for coord in point] for point in bbox_list]
        
        if 'polygon_features' in feature:
            for polygon_feature in feature['polygon_features']['features']:
                if rounding_off==True:
                    int_coords = [[[int(coord) for coord in point] for point in polygon] for polygon in polygon_feature['geometry']['coordinates']]
                    polygon_feature['geometry']['coordinates'] = int_coords
                else:
                    rounded_coords = [[[round(coord, 3) for coord in point] for point in polygon] for polygon in polygon_feature['geometry']['coordinates']]
                    polygon_feature['geometry']['coordinates'] = rounded_coords
    
    return combined_features



def recursive_updating_none(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = recursive_updating_none(value)
    elif isinstance(obj, list):
        obj = [recursive_updating_none(item) for item in obj]
    elif obj == "":
        return None
    return obj


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

        this_description = testing_string

        b_epoch = chronology_period.shape[0]
        t_epoch = -1

        epoch_check = np.flatnonzero(np.core.defchararray.find(this_description, chronology_period)!=-1)
        #print(epoch_check)
        if epoch_check.shape[0] > 0:
            b_epoch = max(epoch_check)
            t_epoch = min(epoch_check)
        
        epoch_check = np.flatnonzero(np.core.defchararray.find(testing_string, chronology_epoch)!=-1)
        #print(epoch_check)
        if epoch_check.shape[0] > 0:
            b_epoch_temp = b_epoch
            if max(epoch_check) <= b_epoch and max(epoch_check) >= t_epoch:
                b_epoch = min(b_epoch ,max(epoch_check))
            else:
                b_epoch = max(b_epoch, max(epoch_check))
            if min(epoch_check) <= b_epoch_temp and min(epoch_check) >= t_epoch:
                t_epoch = max(t_epoch, min(epoch_check))
            else:
                t_epoch = min(t_epoch, min(epoch_check))

        epoch_check = np.flatnonzero(np.core.defchararray.find(testing_string, chronology_age)!=-1)
        #print(epoch_check)
        if epoch_check.shape[0] > 0:
            b_epoch_temp = b_epoch
            if max(epoch_check) <= b_epoch and max(epoch_check) >= t_epoch:
                b_epoch = min(b_epoch ,max(epoch_check))
            else:
                b_epoch = max(b_epoch, max(epoch_check))
            if min(epoch_check) <= b_epoch_temp and min(epoch_check) >= t_epoch:
                t_epoch = max(t_epoch, min(epoch_check))
            else:
                t_epoch = min(t_epoch, min(epoch_check))
        

        #print(testing_string,b_epoch, t_epoch, b_interval, t_interval, b_age, t_age)
        if b_epoch != chronology_period.shape[0] and t_epoch != -1:
            b_interval = chronology_age[b_epoch]
            t_interval = chronology_age[t_epoch]
            b_age = float(chronology_age_b_int[b_epoch])
            t_age = float(chronology_age_t_int[t_epoch])
    
    in_path = os.path.join(dir_to_raster_polygon, fname)
    base_image = cv2.imread(in_path)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    out_path = os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '.geojson'))

    # Open the binary raster file
    src_ds = gdal.Open(in_path)
    src_band = src_ds.GetRasterBand(1)

    # Create an in-memory binary mask where pixel value = 255
    mask_ds = gdal.GetDriverByName('MEM').Create('', src_ds.RasterXSize, src_ds.RasterYSize, 1)
    mask_band = mask_ds.GetRasterBand(1)
    raster_data = src_band.ReadAsArray()
    mask_data = np.where(raster_data == 255, 1, 0).astype(np.uint8)
    mask_band.WriteArray(mask_data)
    mask_band.FlushCache()

    # Prepare output vector (GeoJSON)
    dst_layername = 'polygon'
    drv = ogr.GetDriverByName('geojson')
    dst_ds = drv.CreateDataSource(out_path)

    sp_ref = osr.SpatialReference()
    sp_ref.SetFromUserInput('EPSG:3857')

    dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )
    gdal.Polygonize(src_band, mask_band, dst_layer, 0, [], callback=None )

    del src_ds
    del dst_ds



    try:
        polygon_extraction = gpd.read_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '.geojson')), driver='GeoJSON')
        mirrored_polygon = polygon_extraction.copy()

        debugging_for_unreproducible_error = False

        # Simplify the vectorization results
        mirrored_polygon = mirrored_polygon[mirrored_polygon['geometry'].area/ 10**6 > 0.001]
        mirrored_polygon['id'] = range(0, mirrored_polygon.shape[0])

        if debugging_for_unreproducible_error:
            mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
            mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_0', fname.replace('_predict.png', '_PolygonFeature.geojson')), driver='GeoJSON')

        for index, poi in mirrored_polygon.iterrows():
            this_reduced_polygon = poi['geometry'].simplify(0.9, preserve_topology=True)
            this_reduced_polygon = this_reduced_polygon.buffer(2).buffer(-4).buffer(2)
            this_reduced_polygon = this_reduced_polygon.simplify(0.9, preserve_topology=True)
            #this_reduced_polygon = shapely.wkt.loads(shapely.wkt.dumps(this_reduced_polygon, trim=True, rounding_precision=0))

            mirrored_polygon.loc[index, 'geometry'] = this_reduced_polygon

            if debugging_for_unreproducible_error and index == 0:
                f = open(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_0', fname.replace('_predict.png', '_PolygonFeature_breakdown.txt')), 'w')
                f.write(fname.replace('_predict.png', ''))
                f.write('\n\npoi["geometry"]: \n' + str(poi['geometry']))
                this_reduced_polygon = poi['geometry'].simplify(0.9, preserve_topology=True)
                f.write('\n\nthis_reduced_polygon_v1: \n' + str(this_reduced_polygon))
                this_reduced_polygon = this_reduced_polygon.buffer(2).buffer(-4).buffer(2)
                f.write('\n\nthis_reduced_polygon_v2: \n' + str(this_reduced_polygon))
                this_reduced_polygon = this_reduced_polygon.simplify(0.9, preserve_topology=True)
                f.write('\n\nthis_reduced_polygon_v3: \n' + str(this_reduced_polygon))
                #this_reduced_polygon = shapely.wkt.loads(shapely.wkt.dumps(this_reduced_polygon, trim=True, rounding_precision=0))
                #f.write('\n\nthis_reduced_polygon_v4: \n' + str(this_reduced_polygon))
                f.close()

        
        if debugging_for_unreproducible_error:
            mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
            mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_1', fname.replace('_predict.png', '_PolygonFeature.geojson')), driver='GeoJSON')

        mirrored_polygon = mirrored_polygon.explode()
        mirrored_polygon = mirrored_polygon[mirrored_polygon['geometry'].area/ 10**6 > 0.001]
        mirrored_polygon['id'] = range(0, mirrored_polygon.shape[0])
        
        if debugging_for_unreproducible_error:
            mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
            mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, 'temp_2', fname.replace('_predict.png', '_PolygonFeature.geojson')), driver='GeoJSON')

        #mirrored_polygon = mirrored_polygon.drop('level_0', axis=1)
        #mirrored_polygon = mirrored_polygon.drop('level_1', axis=1)
        mirrored_polygon.reset_index(inplace=True)
        mirrored_polygon = mirrored_polygon.drop('level_0', axis=1)
        mirrored_polygon = mirrored_polygon.drop('level_1', axis=1)
        #print(mirrored_polygon.keys()) 

        # add information
        mirrored_polygon['model'] = ['UMN_USC_Inferlink_Polygon_Extract' for _ in range(mirrored_polygon.shape[0])]
        mirrored_polygon['model_version'] = ['v0.1' for _ in range(mirrored_polygon.shape[0])]
        mirrored_polygon['confidence'] = [0.809 for _ in range(mirrored_polygon.shape[0])]
        mirrored_polygon['model_config'] = [None for _ in range(mirrored_polygon.shape[0])]

        mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
        mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '_v2.geojson')), driver='GeoJSON')





        mirrored_polygon = gpd.GeoDataFrame(columns=['id', 'name', 'geometry', 
                                                        'PolygonType', #: {'id', 'name', 'color', 'pattern', 'abbreviation', 'description', 'category'},  
                                                        'GeologicUnit'#: {'name', 'description', 'comments', 'age_text', 't_interval', 'b_interval', 't_age', 'b_age', 'lithology'}
                                                        ], crs=targeted_crs)
        polygon_extraction = gpd.read_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '_v2.geojson')), driver='GeoJSON')
        
        keys = ["id", "crs", "cdr_projection_id", "map_unit", "abbreviation", "legend_bbox", "description",  "pattern", "color", "category", "polygon_features"]

        mirrored_polygon = pd.DataFrame(columns=keys)
        mirrored_polygon = polygon_extraction.copy()
        mirrored_polygon['id'] = range(0, polygon_extraction.shape[0])
        #mirrored_polygon.drop(mirrored_polygon[mirrored_polygon['id'] == (mirrored_polygon.shape[0]-1)].index, inplace = True)
        mirrored_polygon['crs'] = ['CRITICALMAAS:pixel, EPSG:3857' for _ in range(polygon_extraction.shape[0])]
        mirrored_polygon['cdr_projection_id'] = [None for _ in range(polygon_extraction.shape[0])]
        if info_for_this_poly.shape[0] > 0:
            mirrored_polygon['legend_bbox'] = [str(info_for_this_poly['geometry'].values[0]) for _ in range(polygon_extraction.shape[0])]
        
        if info_for_this_poly.shape[0] != 1:
            mirrored_polygon['map_unit'] = [{'name':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None} for _ in range(polygon_extraction.shape[0])]
            mirrored_polygon['abbreviation'] = [None for _ in range(polygon_extraction.shape[0])]
            mirrored_polygon['description'] = [None for _ in range(polygon_extraction.shape[0])]
            mirrored_polygon['pattern'] = [None for _ in range(polygon_extraction.shape[0])]
            mirrored_polygon['color'] = [None for _ in range(polygon_extraction.shape[0])]
            mirrored_polygon['category'] = [None for _ in range(polygon_extraction.shape[0])]
        else:
            mirrored_polygon['pattern'] = [str(info_for_this_poly['pattern'].values[0]) for _ in range(polygon_extraction.shape[0])]
            mirrored_polygon['color'] = [str(info_for_this_poly['color'].values[0]) for _ in range(polygon_extraction.shape[0])]
            mirrored_polygon['category'] = [str(info_for_this_poly['category'].values[0]) for _ in range(polygon_extraction.shape[0])]
            
            if b_epoch != chronology_period.shape[0]:
                mirrored_polygon['map_unit'] = [{'name':None, 'comments':None, 'age_text':str(b_interval)+' - '+str(t_interval), 't_interval':str(t_interval), 'b_interval':str(b_interval), 't_age':float(t_age), 'b_age':float(b_age), 'lithology':None} for _ in range(polygon_extraction.shape[0])]

                if get_referenced_text == True:
                    mirrored_polygon['abbreviation'] = [str(candidate_info[linking_ids[int(this_abbr)]][1]) for _ in range(polygon_extraction.shape[0])]
                    mirrored_polygon['description'] = [str(candidate_info[linking_ids[int(this_abbr)]][0]) for _ in range(polygon_extraction.shape[0])]
                else:
                    mirrored_polygon['abbreviation'] = [str(info_for_this_poly['abbreviation'].values[0]) for _ in range(polygon_extraction.shape[0])]
                    mirrored_polygon['description'] = [str(info_for_this_poly['description'].values[0]) for _ in range(polygon_extraction.shape[0])]
            else:
                mirrored_polygon['map_unit'] = [{'name':None, 'comments':None, 'age_text':None, 't_interval':None, 'b_interval':None, 't_age':None, 'b_age':None, 'lithology':None} for _ in range(polygon_extraction.shape[0])]

                if get_referenced_text == True:
                    mirrored_polygon['abbreviation'] = [str(candidate_info[linking_ids[int(this_abbr)]][1]) for _ in range(polygon_extraction.shape[0])]
                    mirrored_polygon['description'] = [str(candidate_info[linking_ids[int(this_abbr)]][0]) for _ in range(polygon_extraction.shape[0])]
                else:
                    mirrored_polygon['abbreviation'] = [str(info_for_this_poly['abbreviation'].values[0]) for _ in range(polygon_extraction.shape[0])]
                    mirrored_polygon['description'] = [str(info_for_this_poly['description'].values[0]) for _ in range(polygon_extraction.shape[0])]


        mirrored_polygon.to_file(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '_v3.geojson')), driver='GeoJSON')



        with open(os.path.join(dir_to_integrated_output, 'LOAM_LINK_Intermediate', map_name, fname.replace('_predict.png', '_v3.geojson')),) as f:
            gj = json.load(f)
        combined_features = rebuilding_json(gj, this_abbr, rounding_off=False)
        combined_features = recursive_updating_none(combined_features)

        #empty_geojson = polygon_extraction.set_index('id').T.to_dict('dict')
        with open(os.path.join(dir_to_integrated_output, map_name, fname.replace('_predict.png', '_PolygonFeature.geojson')), 'w') as outfile: 
            json.dump(combined_features, outfile)
            
    except:
        print('Error happended due to a large file size...')
        print('The following is to handle fiona.errors.DriverError: Failed to read or write GeoJSON data...')

        if info_for_this_poly.shape[0] > 0:
            if info_for_this_poly.shape[0] == 1:
                empty_geojson = {
                    "id": "polygon_feature_empty",
                    "crs": "CRITICALMAAS:pixel, EPSG:3857",
                    "cdr_projection_id": None,
                    "legend_bbox": str(info_for_this_poly['geometry'].values[0]),
                    "pattern": str(info_for_this_poly['pattern'].values[0]),
                    "color": str(info_for_this_poly['color'].values[0]),
                    "category": str(info_for_this_poly['category'].values[0]),
                    "map_unit": {
                        "name": None,
                        "comments": None,
                        "age_text": None,
                        "t_interval": None,
                        "b_interval": None,
                        "t_age": None,
                        "b_age": None,
                        "lithology": None
                    },
                    "abbreviation": None,
                    "description": None,
                    "polygon_features": {
                        "features": [{
                                "id": "polygon_feature_empty",
                                "geometry": {
                                    "coordinates": []
                                },
                                "properties": {
                                    "model": "UMN_USC_Inferlink_Polygon_Extract",
                                    "model_version": "v0.1",
                                    "confidence": 0.0,
                                    "model_config": None
                                }
                            }
                        ]
                    }
                }
            else:
                empty_geojson = {
                    "id": "polygon_feature_empty",
                    "crs": "CRITICALMAAS:pixel, EPSG:3857",
                    "cdr_projection_id": None,
                    "legend_bbox": str(info_for_this_poly['geometry'].values[0]),
                    "pattern": None,
                    "color": None,
                    "category": None,
                    "map_unit": {
                        "name": None,
                        "comments": None,
                        "age_text": None,
                        "t_interval": None,
                        "b_interval": None,
                        "t_age": None,
                        "b_age": None,
                        "lithology": None
                    },
                    "abbreviation": None,
                    "description": None,
                    "polygon_features": {
                        "features": [{
                                "id": "polygon_feature_empty",
                                "geometry": {
                                    "coordinates": []
                                },
                                "properties": {
                                    "model": "UMN_USC_Inferlink_Polygon_Extract",
                                    "model_version": "v0.1",
                                    "confidence": 0.0,
                                    "model_config": None
                                }
                            }
                        ]
                    }
                }
        else:
            empty_geojson = {
                "id": "polygon_feature_empty",
                "crs": "CRITICALMAAS:pixel, EPSG:3857",
                "cdr_projection_id": None,
                "legend_bbox": None,
                "pattern": None,
                "color": None,
                "category": None,
                "map_unit": {
                    "name": None,
                    "comments": None,
                    "age_text": None,
                    "t_interval": None,
                    "b_interval": None,
                    "t_age": None,
                    "b_age": None,
                    "lithology": None
                },
                "abbreviation": None,
                "description": None,
                "polygon_features": {
                    "features": [{
                            "id": "polygon_feature_empty",
                            "geometry": {
                                "coordinates": []
                            },
                            "properties": {
                                "model": "UMN_USC_Inferlink_Polygon_Extract",
                                "model_version": "v0.1",
                                "confidence": 0.0,
                                "model_config": None
                            }
                        }
                    ]
                }
            }

        with open(os.path.join(dir_to_integrated_output, map_name, fname.replace('_predict.png', '_PolygonFeature.geojson')), 'w') as outfile:
            json.dump(empty_geojson, outfile)




    return True


