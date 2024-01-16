
from shapely import geometry 
from osgeo import ogr, gdal, osr
from geopandas import GeoDataFrame
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import cv2
import shapely.wkt

import warnings
warnings.filterwarnings("ignore")

def json_generator(binary_image_path, output_path, intermediate_path):
    page_extraction_json = gpd.GeoDataFrame(columns=['name', 'ocr_text', 'color_estimation', 'bounds', 'model', 'geometry'], crs='epsg:3857')


    in_path = os.path.join(intermediate_path, binary_image_path)
    base_image = cv2.imread(in_path)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    out_path = output_path

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

    map_area_detection = gpd.read_file(output_path, driver='GeoJSON')


    map_area_detection['area'] = map_area_detection.geometry.area
    for index, poi in map_area_detection.iterrows():
        #this_area_bounds_cand = poi['geometry'].bounds
        this_area_bounds = shapely.wkt.loads(str(poi['geometry']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
        break
    #this_area_bounds = [this_area_bounds_cand[0], -this_area_bounds_cand[1], this_area_bounds_cand[2], -this_area_bounds_cand[3]]
    #print(this_area_bounds)

    updated_record = gpd.GeoDataFrame([{'name':'map_content', 'ocr_text':'N.A.', 'color_estimation':None, 'bounds': str(this_area_bounds), 'model':None, 'geometry': this_area_bounds}])
    page_extraction_json = gpd.GeoDataFrame(pd.concat( [page_extraction_json, updated_record], ignore_index=True), crs='epsg:3857')

    #print(page_extraction_json)
    page_extraction_json = page_extraction_json.set_crs('epsg:3857', allow_override=True)
    page_extraction_json.to_file(output_path, driver='GeoJSON')

    return True

