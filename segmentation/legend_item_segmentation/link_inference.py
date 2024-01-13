
import numpy as np
from matplotlib.colors import rgb2hex

import os
import cv2

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

from geopandas import GeoDataFrame
from shapely import geometry
from shapely.geometry import Polygon
import shapely.wkt

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.mask import mask

from shapely import affinity

from osgeo import ogr, gdal, osr

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


import shutil
import pyproj
import map_area_segmenter

import json


def processing_uncharted_json(input_legend_segmentation, target_map_name, output_segmentation):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    target_id = -1
    for this_gj in gj['images']:
        this_id = this_gj['id']
        this_file_name = this_gj['file_name']
        if target_map_name.split('.')[0] in this_file_name:
            target_id = this_id
            legend_area_placeholder = np.zeros((this_gj['height'], this_gj['width']), dtype='uint8')
            break
    
    segmentation_added = 0
    for this_gj in gj['annotations']:
        if this_gj['image_id'] == target_id:
            if this_gj['category_id'] == 1 or this_gj['category_id'] == 0:
                coord_counter = 0
                poly_coord = []
                this_coord = []
                for value in this_gj['segmentation'][0]:
                    this_coord.append(int(value))
                    if coord_counter % 2 == 1:
                        this_coord = np.array(this_coord)
                        poly_coord.append(this_coord)
                        this_coord = []
                    coord_counter += 1
                poly_coord = np.array(poly_coord)

                cv2.fillConvexPoly(legend_area_placeholder, poly_coord, 1)
                legend_area_placeholder[legend_area_placeholder > 0] = 255

                segmentation_added += 1
                if segmentation_added >= 2:
                    break
    
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), legend_area_placeholder)
    return True



                    

def map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping):
    print('Working on input image:', input_image)

    if not os.path.exists(path_to_intermediate):
            os.makedirs(path_to_intermediate)

    output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')

    if len(input_legend_segmentation) > 0:
        # if have legend segmentation result (a binary mask that highlights polygon/line/point legends)...
        print('Step (0/9): Processing the given map legend segmentation (whole area) result from... '+str(input_legend_segmentation))
        #print('*Please note that the output json for map area segmentation has not directly used this source...')
        if '.tif' not in input_legend_segmentation:
            print('    Input for legend_area segmentation is not given as a single tif file; will process the json file first...')
            processing_uncharted_json(input_legend_segmentation, target_map_name, output_segmentation)
        else:
            shutil.copyfile(input_legend_segmentation, output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'))
        solution = cv2.imread(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'))
        solution = cv2.cvtColor(solution, cv2.COLOR_BGR2GRAY)
    else:
        if preprocessing_for_cropping == True:
            # if preprocessing for map area segmentation is needed...
            print('Step (0/9): Preprocessing for map area segmentation...')
            map_area_segmenter.cropping_worker(input_image, output_segmentation, path_to_intermediate)
        else:
            # if have area segmentation result...
            print('Step (0/9): Processing the given map area segmentation result from...'+str(input_area_segmentation))
            shutil.copyfile(input_area_segmentation, output_segmentation)
        
        solution = cv2.imread(output_segmentation)
        solution = cv2.cvtColor(solution, cv2.COLOR_BGR2GRAY)
        solution = 255 - solution
        cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), solution)

    img = cv2.imread(input_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.bitwise_and(img, img, mask = solution)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_rgb.tif'), img)

    return True



def text_spotting_with_pytesseract(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, 'intermediate9')):
        os.makedirs(os.path.join(path_to_intermediate, 'intermediate9'))
    print('Step (1/9): Text spotting with pytesseract...')


    map_name = target_map_name
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    base_image2 = cv2.imread(basemap_name)

    img = np.copy(base_image)
    data = pytesseract.image_to_data(img, output_type='data.frame')


    data.drop(data[data['conf'] < 0].index, inplace = True)
    text_geometry = []

    for index, poi in data.iterrows():
        (x, y, w, h) = (poi['left'], poi['top'], poi['width'], poi['height'])
        #Draw box        
        cv2.rectangle(base_image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        xmin = x
        ymin = y
        xmax = x+w
        ymax = y+h

        text_geometry.append(geometry.Polygon(((xmin,ymin), (xmin,ymax), (xmax,ymax), (xmax,ymin))))

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate9', map_name.replace('.tif', '_tesseract.tif'))
    cv2.imwrite(out_file_path0, base_image2)

    data2 = data.assign(geometry = text_geometry)
    data2 = data2.drop(columns=['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height'])
    #geometry0 = data2['geometry'].map(shapely.wkt.loads)
    data3 = GeoDataFrame(data2, crs="EPSG:3857", geometry=data2['geometry'])
    data3.to_file(os.path.join(path_to_intermediate, 'intermediate9', map_name.replace('.tif', '_tesseract.geojson')), driver='GeoJSON')


    ###
    base_image2 = cv2.imread(basemap_name)
    text_geometry = []
    data1 = data.copy()

    for index, poi in data.iterrows():
        (x, y, w, h) = (poi['left'], poi['top'], poi['width'], poi['height'])

        # Modify for a strictor setting
        if w > h:
            if int(h*0.6) > 0:
                y = y+int(h*0.2)
                h = int(h*0.6)
        
        if h > w*2:
            data1 = data1.drop(index)
            continue
        if h > 30:
            data1 = data1.drop(index)
            continue
        #Draw box        
        cv2.rectangle(base_image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        xmin = x
        ymin = y
        xmax = x+w
        ymax = y+h

        text_geometry.append(geometry.Polygon(((xmin,ymin), (xmin,ymax), (xmax,ymax), (xmax,ymin))))

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate9', map_name.replace('.tif', '_tesseract_v2.tif'))
    cv2.imwrite(out_file_path0, base_image2)


    data2 = data1.assign(geometry = text_geometry)
    data2 = data2.drop(columns=['level', 'page_num', 'block_num', 'par_num', 'line_num', 'word_num', 'left', 'top', 'width', 'height'])
    #geometry0 = data2['geometry'].map(shapely.wkt.loads)
    data3 = GeoDataFrame(data2, crs="EPSG:3857", geometry=data2['geometry'])
    data3.to_file(os.path.join(path_to_intermediate, 'intermediate9', map_name.replace('.tif', '_tesseract_v2.geojson')), driver='GeoJSON')

    return True



def read_results_from_pytesseract(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate3'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate3')))
    print('Step (2/9): Processing results from pytesseract...')


    map_name = target_map_name
    mapkurator_source_name = os.path.join(path_to_intermediate, 'intermediate9', map_name.replace('.tif', '_tesseract_v2.geojson'))
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    mapkurator_name = mapkurator_source_name
    gdf = gpd.read_file(mapkurator_name, driver='GeoJSON')
    gdf['area'] = gdf.geometry.area

    text_mask = np.zeros((base_image.shape[0], base_image.shape[1]), dtype='uint8')

    for index, poi in gdf.iterrows():
        with rasterio.open(basemap_name) as src:
            out_image, out_transform = mask(src, [gdf.loc[index]['geometry']], crop=True)
            out_meta = src.meta.copy() # copy the metadata of the source DEM
            
        out_meta.update({
            "driver":"Gtiff",
            "height":out_image.shape[1], # height starts with shape[1]
            "width":out_image.shape[2], # width starts with shape[2]
            "transform":out_transform
        })

        if np.unique(out_image).shape[0] == 1:
            continue
        if poi['area'] > (base_image.shape[0] * base_image.shape[1]) * 0.05:
            continue

        this_text_mask = rasterio.features.rasterize([gdf.loc[index]['geometry']], out_shape=(base_image.shape[0], base_image.shape[1]))
        text_mask = cv2.bitwise_or(text_mask, this_text_mask)
        

    text_mask[text_mask > 0] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask.tif'))
    cv2.imwrite(out_file_path0, text_mask)


    text_mask_buffer = np.copy(text_mask)
    kernel = np.ones((2, 1), np.uint8) # (5, 1)
    text_mask_buffer = cv2.erode(text_mask_buffer, kernel, iterations = 1)
    kernel = np.ones((1, 40), np.uint8)
    text_mask_buffer = cv2.dilate(text_mask_buffer, kernel, iterations = 1)
    text_mask_buffer_temp = np.copy(text_mask_buffer)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v1.tif'))
    cv2.imwrite(out_file_path0, text_mask_buffer)


    kernel = np.ones((1, 500), np.uint8)
    text_mask_buffer = cv2.erode(text_mask_buffer, kernel, iterations = 1)
    kernel = np.ones((1, 500), np.uint8)
    text_mask_buffer = cv2.dilate(text_mask_buffer, kernel, iterations = 1)

    kernel = np.ones((40, 40), np.uint8)
    text_mask_buffer_temp_v0 = cv2.dilate(text_mask_buffer, kernel, iterations = 1)
    small_text_mask_buffer = cv2.bitwise_and(text_mask_buffer_temp, (255-text_mask_buffer_temp_v0))
    for i in range(15):
        small_text_mask_buffer[0:-1, :] = small_text_mask_buffer[1:, :]
    for i in range(20):
        text_mask_buffer[1:, :] = np.maximum(text_mask_buffer[1:, :], text_mask_buffer[0:-1, :]) # growing downwards

    text_mask_buffer = cv2.bitwise_or(text_mask_buffer, text_mask_buffer_temp)
    text_mask_buffer = cv2.bitwise_or(text_mask_buffer, small_text_mask_buffer)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v2.tif'))
    cv2.imwrite(out_file_path0, text_mask_buffer)


    in_path = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v1.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v1.geojson'))

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



    in_path = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v2.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v2.geojson'))

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



    layer1 = gpd.read_file(mapkurator_name, driver='GeoJSON')
    layer2 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v1.geojson')), driver='GeoJSON')

    layer1['x_min'] = layer1.geometry.bounds['minx']
    #layer1['x_max'] = layer1.geometry.bounds['maxx']
    #layer1['y_min'] = layer1.geometry.bounds['miny']
    #layer1['y_min'] = layer1.geometry.bounds['maxy']

    layer2['g_id'] = range(1, layer2.shape[0]+1)
    layer2['g_geometry'] = layer2['geometry']


    overlay_polygon = gpd.overlay(layer1, layer2, how='intersection', keep_geom_type=False)
    text_per_line = gpd.GeoDataFrame(columns=['text', 'geometry'], crs=layer2.crs)

    for gid in range(1, overlay_polygon.shape[0]+1):
        this_gid = overlay_polygon[(overlay_polygon['g_id']==gid)]

        this_gid = this_gid.sort_values(by=['x_min'], ascending=True)
        candidate_text = ''
        for index_this_gid, row_this_gid in this_gid.iterrows():
            if len(candidate_text) == 0:
                candidate_text = str(row_this_gid['text'])
            else:
                candidate_text = candidate_text + ' ' + str(row_this_gid['text'])

        if len(layer2[layer2['g_id']==gid]['g_geometry'].values) == 0:
            continue
        if len(candidate_text) > 500:
            continue
        updated_record = gpd.GeoDataFrame([{'text':candidate_text, 'geometry':layer2[layer2['g_id']==gid]['g_geometry'].values[0]}])
        text_per_line = gpd.GeoDataFrame(pd.concat( [text_per_line, updated_record], ignore_index=True), crs=layer2.crs)

    text_per_line.to_file(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_description_v1.geojson')), driver='GeoJSON')


    layer3 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_description_v1.geojson')), driver='GeoJSON')
    layer4 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v2.geojson')), driver='GeoJSON')

    layer3['l_geometry'] = layer3['geometry']
    layer3['y_min'] = layer3.geometry.bounds['miny']

    layer4['g2_id'] = range(1, layer4.shape[0]+1)
    layer4['g2_geometry'] = layer4['geometry']

    overlay_polygon = gpd.overlay(layer3, layer4, how='intersection', keep_geom_type=False)
    overlay_polygon.drop(overlay_polygon[overlay_polygon.geometry.geometry.type!='Polygon'].index, inplace = True)
    text_combined_line = gpd.GeoDataFrame(columns=['text', 'geometry'], crs=layer4.crs)


    for gid in range(1, overlay_polygon.shape[0]+1):
        this_gid = overlay_polygon[(overlay_polygon['g2_id']==gid)]

        this_gid = this_gid.sort_values(by=['y_min'], ascending=True)
        candidate_text = ''
        for index_this_gid, row_this_gid in this_gid.iterrows():
            if len(candidate_text) == 0:
                candidate_text = str(row_this_gid['text'])
            else:
                candidate_text = candidate_text + ' ' + str(row_this_gid['text'])

        if len(layer4[layer4['g2_id']==gid]['g2_geometry'].values) == 0:
            continue
        if len(candidate_text) > 500:
            continue
        updated_record = gpd.GeoDataFrame([{'text':candidate_text, 'geometry':layer4[layer4['g2_id']==gid]['g2_geometry'].values[0]}])
        text_combined_line = gpd.GeoDataFrame(pd.concat( [text_combined_line, updated_record], ignore_index=True), crs=layer4.crs)

    text_combined_line.to_file(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_description_v2.geojson')), driver='GeoJSON')

    return True





def read_results_from_mapkurator(target_map_name, input_image, path_to_intermediate, path_to_mapkurator_output):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate3_2'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate3_2')))
    print('Step (3/9): Processing results from mapkurator...')


    map_name = target_map_name
    #mapkurator_source_name = os.path.join(path_to_mapkurator_output, map_name.replace('.tif', '.geojson'))
    mapkurator_source_name = path_to_mapkurator_output
    mapkurator_name = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_v2.geojson'))
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)


    with open(mapkurator_source_name, 'r') as file:
        source_data = file.read().replace('-', '')
    with open(mapkurator_name, 'w') as file:
        file.write(source_data)

    gdf = gpd.read_file(mapkurator_name, driver='GeoJSON')
    text_mask = np.zeros((base_image.shape[0], base_image.shape[1]), dtype='uint8')

    for index, poi in gdf.iterrows():
        with rasterio.open(basemap_name) as src:
            out_image, out_transform = mask(src, [gdf.loc[index]['geometry']], crop=True)
            out_meta = src.meta.copy() # copy the metadata of the source DEM
            
        out_meta.update({
            "driver":"Gtiff",
            "height":out_image.shape[1], # height starts with shape[1]
            "width":out_image.shape[2], # width starts with shape[2]
            "transform":out_transform
        })

        if np.unique(out_image).shape[0] == 1:
            continue

        this_text_mask = rasterio.features.rasterize([gdf.loc[index]['geometry']], out_shape=(base_image.shape[0], base_image.shape[1]))
        text_mask = cv2.bitwise_or(text_mask, this_text_mask)
        

    text_mask[text_mask > 0] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask.tif'))
    cv2.imwrite(out_file_path0, text_mask)


    text_mask_buffer = np.copy(text_mask)
    kernel = np.ones((5, 1), np.uint8)
    text_mask_buffer = cv2.erode(text_mask_buffer, kernel, iterations = 1)
    kernel = np.ones((1, 40), np.uint8)
    text_mask_buffer = cv2.dilate(text_mask_buffer, kernel, iterations = 1)
    text_mask_buffer_temp = np.copy(text_mask_buffer)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v1.tif'))
    cv2.imwrite(out_file_path0, text_mask_buffer)


    kernel = np.ones((1, 500), np.uint8)
    text_mask_buffer = cv2.erode(text_mask_buffer, kernel, iterations = 1)
    kernel = np.ones((1, 500), np.uint8)
    text_mask_buffer = cv2.dilate(text_mask_buffer, kernel, iterations = 1)

    kernel = np.ones((40, 40), np.uint8)
    text_mask_buffer_temp_v0 = cv2.dilate(text_mask_buffer, kernel, iterations = 1)
    small_text_mask_buffer = cv2.bitwise_and(text_mask_buffer_temp, (255-text_mask_buffer_temp_v0))
    for i in range(15):
        small_text_mask_buffer[0:-1, :] = small_text_mask_buffer[1:, :]
    for i in range(15):
        text_mask_buffer[1:, :] = np.maximum(text_mask_buffer[1:, :], text_mask_buffer[0:-1, :]) # growing downwards


    text_mask_buffer = cv2.bitwise_or(text_mask_buffer, text_mask_buffer_temp)
    text_mask_buffer = cv2.bitwise_or(text_mask_buffer, small_text_mask_buffer)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v2.tif'))
    cv2.imwrite(out_file_path0, text_mask_buffer)



    in_path = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v1.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v1.geojson'))

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



    in_path = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v2.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v2.geojson'))

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



    layer1 = gpd.read_file(mapkurator_name, driver='GeoJSON')
    layer2 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v1.geojson')), driver='GeoJSON')

    layer1['x_min'] = layer1.geometry.bounds['minx']
    #layer1['x_max'] = layer1.geometry.bounds['maxx']
    #layer1['y_min'] = layer1.geometry.bounds['miny']
    #layer1['y_min'] = layer1.geometry.bounds['maxy']

    layer2['g_id'] = range(1, layer2.shape[0]+1)
    layer2['g_geometry'] = layer2['geometry']


    overlay_polygon = gpd.overlay(layer1, layer2, how='intersection', keep_geom_type=False)
    text_per_line = gpd.GeoDataFrame(columns=['text', 'geometry'], crs=layer2.crs)

    for gid in range(1, overlay_polygon.shape[0]+1):
        this_gid = overlay_polygon[(overlay_polygon['g_id']==gid)]

        this_gid = this_gid.sort_values(by=['x_min'], ascending=True)
        candidate_text = ''
        for index_this_gid, row_this_gid in this_gid.iterrows():
            if len(candidate_text) == 0:
                candidate_text = str(row_this_gid['text'])
            else:
                candidate_text = candidate_text + ' ' + str(row_this_gid['text'])

        if len(layer2[layer2['g_id']==gid]['g_geometry'].values) == 0:
            continue
        if len(candidate_text) > 500:
            continue
        updated_record = gpd.GeoDataFrame([{'text':candidate_text, 'geometry':layer2[layer2['g_id']==gid]['g_geometry'].values[0]}])
        text_per_line = gpd.GeoDataFrame(pd.concat( [text_per_line, updated_record], ignore_index=True), crs=layer2.crs)
    text_per_line.to_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_description_v1.geojson')), driver='GeoJSON')


    layer3 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_description_v1.geojson')), driver='GeoJSON')
    layer4 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v2.geojson')), driver='GeoJSON')

    layer3['l_geometry'] = layer3['geometry']
    layer3['y_min'] = layer3.geometry.bounds['miny']

    layer4['g2_id'] = range(1, layer4.shape[0]+1)
    layer4['g2_geometry'] = layer4['geometry']

    overlay_polygon = gpd.overlay(layer3, layer4, how='intersection', keep_geom_type=False)
    overlay_polygon.drop(overlay_polygon[overlay_polygon.geometry.geometry.type!='Polygon'].index, inplace = True)
    text_combined_line = gpd.GeoDataFrame(columns=['text', 'geometry'], crs=layer4.crs)



    for gid in range(1, overlay_polygon.shape[0]+1):
        this_gid = overlay_polygon[(overlay_polygon['g2_id']==gid)]

        this_gid = this_gid.sort_values(by=['y_min'], ascending=True)
        candidate_text = ''
        for index_this_gid, row_this_gid in this_gid.iterrows():
            if len(candidate_text) == 0:
                candidate_text = str(row_this_gid['text'])
            else:
                candidate_text = candidate_text + ' ' + str(row_this_gid['text'])

        if len(layer4[layer4['g2_id']==gid]['g2_geometry'].values) == 0:
            continue
        if len(candidate_text) > 500:
            continue
        updated_record = gpd.GeoDataFrame([{'text':candidate_text, 'geometry':layer4[layer4['g2_id']==gid]['g2_geometry'].values[0]}])
        text_combined_line = gpd.GeoDataFrame(pd.concat( [text_combined_line, updated_record], ignore_index=True), crs=layer4.crs)

    text_combined_line.to_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_description_v2.geojson')), driver='GeoJSON')

    return True





def map_key_extraction_polygon(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate4'))):
        os.makedirs(os.path.join(path_to_intermediate,  str('intermediate4')))
    print('Step (4/9): Extracting map keys (symbols) for polygon features...')


    map_name = target_map_name
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    lower_color = np.array([250])
    upper_color = np.array([256])
    base_image_mask_white = cv2.inRange(base_image, lower_color, upper_color)
    base_image_mask_colored = cv2.bitwise_and(base_image, (255-base_image_mask_white))
    base_image_mask_colored[base_image_mask_colored > 0] = 255


    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_colored.tif'))
    cv2.imwrite(out_file_path0, base_image_mask_colored)

    lower_color = np.array([0])
    upper_color = np.array([70])
    base_image_mask_black = cv2.inRange(base_image, lower_color, upper_color)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_black.tif'))
    cv2.imwrite(out_file_path0, base_image_mask_black)


    kernel = np.ones((5, 5), np.uint8)
    base_image_mask_black_buffer = cv2.dilate(base_image_mask_black, kernel, iterations = 1)
    map_key_candidate = cv2.bitwise_and(base_image_mask_colored, (255-base_image_mask_black_buffer))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    opening = cv2.morphologyEx(map_key_candidate, cv2.MORPH_OPEN, kernel, iterations=1)
    map_key_candidate = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

    # flood fill background to find inner holes
    floodfill_candidate = np.ones((base_image.shape[0], base_image.shape[1]), dtype='uint8') * 255
    holes = np.copy(map_key_candidate)
    cv2.floodFill(holes, None, (0, 0), 255)

    # invert holes mask, bitwise or with img fill in holes
    holes = cv2.bitwise_not(holes)
    valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
    map_key_candidate = cv2.bitwise_or(map_key_candidate, valid_holes)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_map_key.tif'))
    cv2.imwrite(out_file_path0, map_key_candidate)
    
    reference_gpd = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v2.geojson')), driver='GeoJSON')
    polygon_candidate_gpd = gpd.GeoDataFrame(columns=['poly_id', 'geometry'], crs=reference_gpd.crs)

    base_image2 = cv2.imread(basemap_name)

    # Find contours and filter using threshold area
    cnts = cv2.findContours(map_key_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    image_number = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        #ROI = base_image2[y:y+h, x:x+w]
        if (w > h*1.33) and (w < h*4) and (area > 3000.0) and (area < 12000.0):
            cv2.rectangle(base_image2, (x, y), (x + w, y + h), (36,255,12), 4)
            image_number += 1

            this_contour = c[:,0,:]
            this_polygon = Polygon(this_contour)

            updated_record = gpd.GeoDataFrame([{'poly_id':image_number, 'geometry':this_polygon}])
            polygon_candidate_gpd = gpd.GeoDataFrame(pd.concat( [polygon_candidate_gpd, updated_record], ignore_index=True), crs=reference_gpd.crs)

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_map_key_highlighted.tif'))
    cv2.imwrite(out_file_path0, base_image2)


    relaxation = 0
    while image_number == 0:
        lower_color = np.array([0])
        upper_color = np.array([100 + relaxation*10])
        base_image_mask_black = cv2.inRange(base_image, lower_color, upper_color)
        out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_black.tif'))
        cv2.imwrite(out_file_path0, base_image_mask_black)

        kernel = np.ones((7+relaxation, 7+relaxation), np.uint8)
        base_image_mask_black_buffer = cv2.dilate(base_image_mask_black, kernel, iterations = 1)
        map_key_candidate = cv2.bitwise_and(base_image_mask_colored, (255-base_image_mask_black_buffer))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7+relaxation,7+relaxation))
        opening = cv2.morphologyEx(map_key_candidate, cv2.MORPH_OPEN, kernel, iterations=1)
        map_key_candidate = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

        # flood fill background to find inner holes
        floodfill_candidate = np.ones((base_image.shape[0], base_image.shape[1]), dtype='uint8') * 255
        holes = np.copy(map_key_candidate)
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
        map_key_candidate = cv2.bitwise_or(map_key_candidate, valid_holes)
        out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_map_key.tif'))
        cv2.imwrite(out_file_path0, map_key_candidate)

        
        reference_gpd = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_buffer_v2.geojson')), driver='GeoJSON')
        polygon_candidate_gpd = gpd.GeoDataFrame(columns=['poly_id', 'geometry'], crs=reference_gpd.crs)

        base_image2 = cv2.imread(basemap_name)

        # Find contours and filter using threshold area
        cnts = cv2.findContours(map_key_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        image_number = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            #ROI = base_image2[y:y+h, x:x+w]
            if (w > h*1.33) and (w < h*4) and (area > 3000.0) and (area < 12000.0):
                cv2.rectangle(base_image2, (x, y), (x + w, y + h), (36,255,12), 4)
                image_number += 1

                this_contour = c[:,0,:]
                this_polygon = Polygon(this_contour)

                updated_record = gpd.GeoDataFrame([{'poly_id':image_number, 'geometry':this_polygon}])
                polygon_candidate_gpd = gpd.GeoDataFrame(pd.concat( [polygon_candidate_gpd, updated_record], ignore_index=True), crs=reference_gpd.crs)

        out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_map_key_highlighted.tif'))
        cv2.imwrite(out_file_path0, base_image2)

        relaxation = relaxation + 1



    polygon_candidate_gpd.to_file(os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_polygon_candidate_v1.geojson')), driver='GeoJSON')

    polygon_candidate_mask = np.zeros((base_image.shape[0], base_image.shape[1]), dtype='uint8')
    for index, poi in polygon_candidate_gpd.iterrows():
        with rasterio.open(basemap_name) as src:
            out_image, out_transform = mask(src, [polygon_candidate_gpd.loc[index]['geometry']], crop=True)
            out_meta = src.meta.copy() # copy the metadata of the source DEM
            
        out_meta.update({
            "driver":"Gtiff",
            "height":out_image.shape[1], # height starts with shape[1]
            "width":out_image.shape[2], # width starts with shape[2]
            "transform":out_transform
        })

        this_poly_mask = rasterio.features.rasterize([polygon_candidate_gpd.loc[index]['geometry']], out_shape=(base_image.shape[0], base_image.shape[1]))
        polygon_candidate_mask = cv2.bitwise_or(polygon_candidate_mask, this_poly_mask)

    polygon_candidate_mask[polygon_candidate_mask > 0] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_map_key.tif'))
    cv2.imwrite(out_file_path0, polygon_candidate_mask)




    link_seeking = np.copy(polygon_candidate_mask)

    kernel = np.ones((40, 1), np.uint8)
    link_seeking = cv2.erode(link_seeking, kernel, iterations = 1)
    for i in range(150):
        link_seeking[:, 1:] = np.maximum(link_seeking[:, 1:], link_seeking[:, 0:-1]) # growing right

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_link_seeking_v1.tif'))
    cv2.imwrite(out_file_path0, link_seeking)

    in_path = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_link_seeking_v1.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_link_seeking_v1.geojson'))

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

    return True



def linking_description_polygon(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate,  str('intermediate5/'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate5/')))
    print('Step (5/9): Linking symbols to text descriptions for polygon map keys...')


    map_name = target_map_name
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image2 = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)


    layer_polygon_candidate = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_polygon_candidate_v1.geojson')), driver='GeoJSON')
    layer_link_seeking = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_link_seeking_v1.geojson')), driver='GeoJSON')
    layer_mask_description = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_mapkurator_mask_description_v2.geojson')), driver='GeoJSON')

    layer_mask_description_2 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_description_v2.geojson')), driver='GeoJSON')

    #layer_polygon_candidate['poly_id'] = range(1, layer_polygon_candidate.shape[0]+1)
    layer_polygon_candidate['p_geometry'] = layer_polygon_candidate['geometry']
    layer_link_seeking['seeking_id'] = range(1, layer_link_seeking.shape[0]+1)
    layer_link_seeking['s_geometry'] = layer_link_seeking['geometry']
    layer_mask_description['text_id'] = range(1, layer_mask_description.shape[0]+1)
    layer_mask_description['t_geometry'] = layer_mask_description['geometry']
    layer_mask_description['area'] = layer_mask_description.geometry.area
    layer_mask_description.drop(layer_mask_description[layer_mask_description['area'] > (base_image.shape[0]*base_image.shape[1])*0.8].index, inplace = True)

    layer_mask_description_2['text_id'] = range(1, layer_mask_description_2.shape[0]+1)
    layer_mask_description_2['t_geometry'] = layer_mask_description_2['geometry']
    layer_mask_description_2['area'] = layer_mask_description_2.geometry.area
    layer_mask_description_2.drop(layer_mask_description_2[layer_mask_description_2['area'] > (base_image.shape[0]*base_image.shape[1])*0.1].index, inplace = True)
    layer_mask_description_2.drop(layer_mask_description_2[layer_mask_description_2['text'].str.len() > 1000].index, inplace = True)

    overlay_polygon_bridge_1 = gpd.overlay(layer_polygon_candidate, layer_link_seeking, how='intersection', keep_geom_type=False)
    overlay_polygon_bridge_2 = gpd.overlay(layer_mask_description, layer_link_seeking, how='intersection', keep_geom_type=False)

    overlay_polygon_bridge_2_2 = gpd.overlay(layer_mask_description_2, layer_link_seeking, how='intersection', keep_geom_type=False)


    poly_id_to_drop = []
    seeking_id_to_drop = []
    # if multiple polygon candidates (map keys) are in the same link-seeking group, exclude them
    for sid in range(1, overlay_polygon_bridge_1.shape[0]+1):
        this_sid = overlay_polygon_bridge_1[(overlay_polygon_bridge_1['seeking_id']==sid)]

        if this_sid.shape[0] == layer_polygon_candidate.shape[0]:
            seeking_id_to_drop.append(sid)
        elif this_sid.shape[0] > 1:
            for index_this_sid, row_this_sid in this_sid.iterrows():
                poly_id_to_drop.append(row_this_sid['poly_id'])
                seeking_id_to_drop.append(row_this_sid['seeking_id'])
            
    for targeted_poly_id in poly_id_to_drop:
        layer_polygon_candidate.drop(layer_polygon_candidate[layer_polygon_candidate['poly_id'] == targeted_poly_id].index, inplace = True)
        overlay_polygon_bridge_1.drop(overlay_polygon_bridge_1[overlay_polygon_bridge_1['poly_id'] == targeted_poly_id].index, inplace = True)
    for targeted_seeking_id in seeking_id_to_drop:
        layer_link_seeking.drop(layer_link_seeking[layer_link_seeking['seeking_id'] == targeted_seeking_id].index, inplace = True)
        overlay_polygon_bridge_1.drop(overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id'] == targeted_seeking_id].index, inplace = True)
        overlay_polygon_bridge_2.drop(overlay_polygon_bridge_2[overlay_polygon_bridge_2['seeking_id'] == targeted_seeking_id].index, inplace = True)

        overlay_polygon_bridge_2_2.drop(overlay_polygon_bridge_2_2[overlay_polygon_bridge_2_2['seeking_id'] == targeted_seeking_id].index, inplace = True)


    

    linked_poly_description = gpd.GeoDataFrame(columns=['poly_id', 'seeking_id', 'text_id1', 'text_id2', 'poly_geo', 'text_geo1', 'text_geo2', 'text_description1', 'text_description2', 'geometry'], crs=layer_mask_description.crs)
    for index, row in overlay_polygon_bridge_1.iterrows():
        this_poly = overlay_polygon_bridge_1[(overlay_polygon_bridge_1['seeking_id'] == row['seeking_id'])]
        this_text = overlay_polygon_bridge_2[(overlay_polygon_bridge_2['seeking_id'] == row['seeking_id'])]
        this_text_2 = overlay_polygon_bridge_2_2[(overlay_polygon_bridge_2_2['seeking_id'] == row['seeking_id'])]

        this_text = this_text.sort_values(by=['area'], ascending=False)
        this_text_2 = this_text_2.sort_values(by=['area'], ascending=False)

        poly_id = int(this_poly['poly_id'])
        poly_geo = this_poly['p_geometry']
        seeking_id = int(this_poly['seeking_id'])
        seeking_geo = this_poly['s_geometry']
        text_id_set = []
        text_geo_set = []
        text_description_set = []

        if this_text.shape[0] == 1:
            # There is no text label in the map key
            if this_text_2.shape[0] > 0:
                for index_this_text, row_this_text in this_text_2.iterrows():
                    text_id_set.append(int(row_this_text['text_id']))
                    text_geo_set.append(row_this_text['t_geometry'])
                    text_description_set.append(row_this_text['text'])
                    break
            else:
                for index_this_text, row_this_text in this_text.iterrows():
                    text_id_set.append(int(row_this_text['text_id']))
                    text_geo_set.append(row_this_text['t_geometry'])
                    text_description_set.append(row_this_text['text'])
                    break

            text_id_set = np.array(text_id_set)
            text_geo_set = np.array(text_geo_set)
            text_description_set = np.array(text_description_set)
            
            updated_record = gpd.GeoDataFrame([{'poly_id':poly_id, 'seeking_id':seeking_id, 'text_id1':text_id_set[0], 'text_id2':-1, 'poly_geo':str(poly_geo.values[0]), 'text_geo1':text_geo_set[0].wkt, 'text_geo2':text_geo_set[0].wkt, 'text_description1':text_description_set[0], 'text_description2':'', 'geometry': overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id']==seeking_id].geometry.values[0]}])
        elif this_text.shape[0] > 1:
            # There is text label in the map key
            if this_text_2.shape[0] > 0:
                for index_this_text, row_this_text in this_text_2.iterrows():
                    text_id_set.append(int(row_this_text['text_id']))
                    text_geo_set.append(row_this_text['t_geometry'])
                    text_description_set.append(row_this_text['text'])
                    break
                flag = False
                for index_this_text, row_this_text in this_text.iterrows():
                    if flag == False:
                        flag = True
                        continue
                    text_id_set.append(int(row_this_text['text_id']))
                    text_geo_set.append(row_this_text['t_geometry'])
                    text_description_set.append(row_this_text['text'])
                    break
            else:
                for index_this_text, row_this_text in this_text.iterrows():
                    text_id_set.append(int(row_this_text['text_id']))
                    text_geo_set.append(row_this_text['t_geometry'])
                    text_description_set.append(row_this_text['text'])

            text_id_set = np.array(text_id_set)
            text_geo_set = np.array(text_geo_set)
            text_description_set = np.array(text_description_set)

            updated_record = gpd.GeoDataFrame([{'poly_id':poly_id, 'seeking_id':seeking_id, 'text_id1':text_id_set[0], 'text_id2':text_id_set[1], 'poly_geo':str(poly_geo.values[0]), 'text_geo1':text_geo_set[0].wkt, 'text_geo2':text_geo_set[1].wkt, 'text_description1':text_description_set[0], 'text_description2':text_description_set[1], 'geometry': overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id']==seeking_id].geometry.values[0]}])


        if this_text.shape[0] >= 1:
            linked_poly_description = gpd.GeoDataFrame(pd.concat( [linked_poly_description, updated_record], ignore_index=True), crs=layer_mask_description.crs)

            x1, y1, x2, y2 = overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id']==seeking_id].geometry.values[0].bounds
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(base_image2, (x1, y1), (x2, y2), (22,59,18), 4)
            if this_text.shape[0] == 1:
                cv2.putText(base_image2, str(text_description_set[0]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (22,59,18), 3)
            else:
                cv2.putText(base_image2, str(text_description_set[1] + ': ' + text_description_set[0]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (22,59,18), 3)
        else:
            continue
    linked_poly_description.to_file(os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_legend.geojson')), driver='GeoJSON')

    
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_legend.tif'))
    cv2.imwrite(out_file_path0, base_image2)

    return True



def searching_possible_description(target_map_name, input_image, path_to_intermediate):
    print('Step (6/9): Finding text descriptions possibly for point/ line features...')


    map_name = target_map_name
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')
    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    
    linked_poly_description = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_legend.geojson')), driver='GeoJSON')
    

    approximate_candidate = np.zeros((base_image.shape[0], base_image.shape[1]), dtype='uint8')
    x1, y1, x2, y2 = linked_poly_description.total_bounds
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    approximate_candidate[y1:, x1:] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_approximate_candidate.tif'))
    cv2.imwrite(out_file_path0, approximate_candidate)


    linked_text_description = np.zeros((base_image.shape[0], base_image.shape[1]), dtype='uint8')
    for index, poi in linked_poly_description.iterrows():
        with rasterio.open(basemap_name) as src:
            out_image, out_transform = mask(src, [shapely.wkt.loads(linked_poly_description.loc[index]['text_geo1'])], crop=True)
            out_meta = src.meta.copy() # copy the metadata of the source DEM
            
        out_meta.update({
            "driver":"Gtiff",
            "height":out_image.shape[1], # height starts with shape[1]
            "width":out_image.shape[2], # width starts with shape[2]
            "transform":out_transform
        })

        this_poly_mask = rasterio.features.rasterize([shapely.wkt.loads(linked_poly_description.loc[index]['text_geo1'])], out_shape=(base_image.shape[0], base_image.shape[1]))
        linked_text_description = cv2.bitwise_or(linked_text_description, this_poly_mask)

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_description.tif'))
    cv2.imwrite(out_file_path0, linked_text_description)


    linked_polygon_key = np.zeros((base_image.shape[0], base_image.shape[1]), dtype='uint8')
    for index, poi in linked_poly_description.iterrows():
        with rasterio.open(basemap_name) as src:
            out_image, out_transform = mask(src, [shapely.wkt.loads(linked_poly_description.loc[index]['poly_geo'])], crop=True)
            out_meta = src.meta.copy() # copy the metadata of the source DEM
            
        out_meta.update({
            "driver":"Gtiff",
            "height":out_image.shape[1], # height starts with shape[1]
            "width":out_image.shape[2], # width starts with shape[2]
            "transform":out_transform
        })

        this_poly_mask = rasterio.features.rasterize([shapely.wkt.loads(linked_poly_description.loc[index]['poly_geo'])], out_shape=(base_image.shape[0], base_image.shape[1]))
        linked_polygon_key = cv2.bitwise_or(linked_polygon_key, this_poly_mask)

    linked_polygon_key[linked_polygon_key > 0] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_key.tif'))
    cv2.imwrite(out_file_path0, linked_polygon_key)



    text_description_candidate = cv2.imread(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_buffer_v2.tif')))
    legend_pl_candidate = cv2.imread(os.path.join(path_to_intermediate, 'intermediate4', map_name.replace('.tif', '_mask_colored.tif')))
    text_description_candidate = cv2.cvtColor(text_description_candidate, cv2.COLOR_BGR2GRAY)
    legend_pl_candidate = cv2.cvtColor(legend_pl_candidate, cv2.COLOR_BGR2GRAY)


    text_description_buffer = np.copy(linked_text_description)
    text_description_buffer[text_description_buffer > 0] = 0

    # Find the average of each channel across the image
    mean_per_column = [np.mean(linked_text_description[:,i]) for i in range(linked_text_description.shape[1])]
    mean_per_column = np.array(mean_per_column)
    mean_per_column[mean_per_column > 0] = 255

    # Find additional possible columns for text description
    mean_change_per_column = np.zeros((mean_per_column.shape[0]), dtype='uint8')
    mean_change_per_column[:-1] = mean_per_column[1:] - mean_per_column[:-1]
    mean_change_per_column[mean_change_per_column != 0] = 255
    mean_change_point = np.where(mean_change_per_column > 0)[0]

    column_width_set = []
    column_padding_set = []
    added_column = 0
    if mean_change_point.shape[0] > 0:
        if mean_change_point.shape[0] >= 2:
            for this_change in range(0, mean_change_point.shape[0], 2):
                this_column_width = mean_change_point[this_change+1] - mean_change_point[this_change]
                column_width_set.append(this_column_width)
                if this_change > 0:
                    this_column_padding = mean_change_point[this_change] - mean_change_point[this_change-1]
                    column_padding_set.append(this_column_padding)
        column_width_set = np.array(column_width_set)
        column_padding_set = np.array(column_padding_set)

        
        defined_column_width = np.mean(column_width_set)*0.8
        defined_column_padding = np.mean(column_padding_set)
        defined_columnn_threshold = int(defined_column_width * 0.15)


        for additional_column in range(1, 20):
            this_column_x1 = max(mean_change_point) + defined_column_padding*additional_column + defined_columnn_threshold
            this_column_x2 = max(mean_change_point) + defined_column_padding*additional_column + defined_column_width - defined_columnn_threshold
            if this_column_x2 > mean_per_column.shape[0]:
                break
            mean_per_column[int(this_column_x1): int(this_column_x2)] = 255
            added_column += 1

        if mean_change_point.shape[0] >= 2:
            for this_change in range(0, mean_change_point.shape[0], 2):
                mean_per_column[mean_change_point[this_change]:mean_change_point[this_change]+defined_columnn_threshold] = 0
                mean_per_column[mean_change_point[this_change+1]-defined_columnn_threshold:mean_change_point[this_change+1]+1] = 0



        text_description_buffer[:, np.where(mean_per_column > 0)] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_description_v2.tif'))
    cv2.imwrite(out_file_path0, text_description_buffer)

    text_description_buffer = cv2.bitwise_and(text_description_buffer, text_description_candidate)
    text_description_buffer = cv2.bitwise_and(text_description_buffer, (255-linked_text_description))
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_description_v3.tif'))
    cv2.imwrite(out_file_path0, text_description_buffer)


    in_path = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_description_v3.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_description_v3.geojson'))

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




    poly_key_buffer = np.copy(linked_polygon_key)
    poly_key_buffer[poly_key_buffer > 0] = 0

    # Find the average of each channel across the image
    mean_per_column = [np.mean(linked_polygon_key[:,i]) for i in range(linked_polygon_key.shape[1])]
    mean_per_column = np.array(mean_per_column)
    mean_per_column[mean_per_column > 0] = 255

    # Find additional possible columns for map keys
    mean_change_per_column = np.zeros((mean_per_column.shape[0]), dtype='uint8')
    mean_change_per_column[:-1] = mean_per_column[1:] - mean_per_column[:-1]
    mean_change_per_column[mean_change_per_column != 0] = 255
    mean_change_point = np.where(mean_change_per_column > 0)[0]

    column_width_set = []
    column_padding_set = []
    if mean_change_point.shape[0] > 0:
        if mean_change_point.shape[0] >= 2:
            for this_change in range(0, mean_change_point.shape[0], 2):
                this_column_width = mean_change_point[this_change+1] - mean_change_point[this_change]
                column_width_set.append(this_column_width)
                if this_change > 0:
                    this_column_padding = mean_change_point[this_change] - mean_change_point[this_change-1]
                    column_padding_set.append(this_column_padding)
        column_width_set = np.array(column_width_set)
        column_padding_set = np.array(column_padding_set)

        defined_column_width = np.mean(column_width_set)*0.8
        defined_column_padding = np.mean(column_padding_set)

        for additional_column in range(1, added_column+1):
            this_column_x1 = max(mean_change_point) + defined_column_padding*additional_column
            this_column_x2 = max(mean_change_point) + defined_column_padding*additional_column + defined_column_width
            if this_column_x2 > mean_per_column.shape[0]:
                break
            mean_per_column[int(this_column_x1): int(this_column_x2)] = 255


        poly_key_buffer[:, np.where(mean_per_column > 0)] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_key_v2.tif'))
    cv2.imwrite(out_file_path0, poly_key_buffer)


    kernel = np.ones((50, 200), np.uint8)
    linked_polygon_key = cv2.dilate(linked_polygon_key, kernel, iterations = 1)
    
    poly_key_buffer = cv2.bitwise_and(poly_key_buffer, legend_pl_candidate)
    poly_key_buffer = cv2.bitwise_and(poly_key_buffer, (255-linked_polygon_key))
    poly_key_buffer = cv2.bitwise_and(poly_key_buffer, approximate_candidate)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_key_v3.tif'))
    cv2.imwrite(out_file_path0, poly_key_buffer)

    return True






def linking_description_pointline(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate6'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate6')))
    print('Step (7/9): Linking text descriptions to symbols for point/ line map keys...')


    map_name = target_map_name
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image2 = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    candidate_poly_description_text = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_tesseract_mask_description_v2.geojson')), driver='GeoJSON')
    candidate_poly_description_area = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_description_v3.geojson')), driver='GeoJSON')

    
    candidate_poly_description_text['text_id'] = range(1, candidate_poly_description_text.shape[0]+1)
    #candidate_poly_description_text['t_geometry'] = candidate_poly_description_text['geometry']
    candidate_poly_description_text['t_area'] = candidate_poly_description_text.geometry.area
    candidate_poly_description_text.drop(candidate_poly_description_text[candidate_poly_description_text['t_area'] > (base_image.shape[0]*base_image.shape[1])*0.8].index, inplace = True)
    candidate_poly_description_area['area_id'] = range(1, candidate_poly_description_area.shape[0]+1)
    candidate_poly_description_area['a_area'] = candidate_poly_description_area.geometry.area
    candidate_poly_description_area.drop(candidate_poly_description_area[candidate_poly_description_area['a_area'] > (base_image.shape[0]*base_image.shape[1])*0.8].index, inplace = True)

    overlay_candidate_poly_description = gpd.overlay(candidate_poly_description_text, candidate_poly_description_area, how='intersection', keep_geom_type=False)

    text_id_to_drop = []
    # if multiple polygon candidates (map keys) are in the same link-seeking group, exclude them
    for tid in range(1, overlay_candidate_poly_description.shape[0]+1):
        this_tid = overlay_candidate_poly_description[(overlay_candidate_poly_description['text_id']==tid)]
        if this_tid.shape[0] == 0:
            text_id_to_drop.append(tid)
    for targeted_text_id in text_id_to_drop:
        overlay_candidate_poly_description.drop(overlay_candidate_poly_description[overlay_candidate_poly_description['text_id'] == targeted_text_id].index, inplace = True)
    overlay_candidate_poly_description.to_file(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_polygon_description_v4.geojson')), driver='GeoJSON')

    overlay_candidate_poly_description = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_polygon_description_v4.geojson')), driver='GeoJSON')
    
    linked_text_description = np.zeros((base_image.shape[0], base_image.shape[1]), dtype='uint8')
    for index, poi in overlay_candidate_poly_description.iterrows():
        this_text_id = poi['text_id']
        with rasterio.open(basemap_name) as src:
            out_image, out_transform = mask(src, [candidate_poly_description_text[candidate_poly_description_text['text_id'] == this_text_id]['geometry'].values[0]], crop=True) # [gdf.loc[index].geometry]
            out_meta = src.meta.copy() # copy the metadata of the source DEM
            
        out_meta.update({
            "driver":"Gtiff",
            "height":out_image.shape[1], # height starts with shape[1]
            "width":out_image.shape[2], # width starts with shape[2]
            "transform":out_transform
        })

        this_poly_mask = rasterio.features.rasterize([candidate_poly_description_text[candidate_poly_description_text['text_id'] == this_text_id]['geometry'].values[0]], out_shape=(base_image.shape[0], base_image.shape[1]))
        linked_text_description = cv2.bitwise_or(linked_text_description, this_poly_mask)

    linked_text_description[linked_text_description > 0] = 255
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_polygon_description_v4.tif'))
    cv2.imwrite(out_file_path0, linked_text_description)
    


    link_seeking_re = np.copy(linked_text_description)
    link_seeking_place_holder = np.copy(linked_text_description)
    for i in range(200):
        link_seeking_re[:, :-1] = np.maximum(link_seeking_re[:, 1:], link_seeking_re[:, :-1]) # growing left
    
    kernel = np.ones((40, 20), np.uint8)
    link_seeking_place_holder = cv2.dilate(link_seeking_place_holder, kernel, iterations = 1)
    for i in range(50):
        link_seeking_place_holder[:, 1:] = np.maximum(link_seeking_place_holder[:, 1:], link_seeking_place_holder[:, 0:-1]) # growing right
    
    link_seeking_re = cv2.bitwise_and(link_seeking_re, (255-link_seeking_place_holder))


    link_seeking_bridge = np.copy(link_seeking_re)
    for i in range(250):
        link_seeking_bridge[:, 1:] = np.maximum(link_seeking_bridge[:, 1:], link_seeking_bridge[:, 0:-1]) # growing right

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_linking.tif'))
    cv2.imwrite(out_file_path0, link_seeking_bridge)


    in_path = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_linking.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_linking.geojson'))

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

    

    candidate_key = cv2.imread(os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_candidate_polygon_key_v3.tif')))
    candidate_key = cv2.cvtColor(candidate_key, cv2.COLOR_BGR2GRAY)

    candidate_key = cv2.bitwise_and(candidate_key, link_seeking_re)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_v1.tif'))
    cv2.imwrite(out_file_path0, candidate_key)

    kernel = np.ones((20, 20), np.uint8)
    candidate_key = cv2.dilate(candidate_key, kernel, iterations = 1)
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_v2.tif'))
    cv2.imwrite(out_file_path0, candidate_key)

    in_path = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_v2.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_v2.geojson'))

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


    # ======
    map_name = target_map_name
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image2 = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    layer_pl_candidate = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_v2.geojson')), driver='GeoJSON')
    layer_link_seeking = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_pl_linking.geojson')), driver='GeoJSON')
    layer_mask_description = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_candidate_polygon_description_v4.geojson')), driver='GeoJSON')

    layer_pl_candidate['pl_id'] = range(1, layer_pl_candidate.shape[0]+1)
    layer_pl_candidate['p_geometry'] = layer_pl_candidate['geometry']
    layer_pl_candidate['p_area'] = layer_pl_candidate.geometry.area
    layer_pl_candidate.drop(layer_pl_candidate[layer_pl_candidate['p_area'] > (base_image.shape[0]*base_image.shape[1])*0.8].index, inplace = True)
    layer_link_seeking['seeking_id'] = range(1, layer_link_seeking.shape[0]+1)
    layer_link_seeking['s_geometry'] = layer_link_seeking['geometry']
    layer_link_seeking['l_area'] = layer_link_seeking.geometry.area
    layer_link_seeking.drop(layer_link_seeking[layer_link_seeking['l_area'] > (base_image.shape[0]*base_image.shape[1])*0.8].index, inplace = True)
    
    layer_mask_description.drop(layer_mask_description[layer_mask_description.geometry.geometry.type!='Polygon'].index, inplace = True)
    layer_mask_description.drop('a_area', axis=1, inplace=True)
    layer_mask_description.drop('area_id', axis=1, inplace=True)
    layer_mask_description['text_id'] = range(1, layer_mask_description.shape[0]+1)
    layer_mask_description['t_geometry'] = layer_mask_description['geometry']
    layer_mask_description['t_area'] = layer_mask_description.geometry.area
    layer_mask_description.drop(layer_mask_description[layer_mask_description['t_area'] > (base_image.shape[0]*base_image.shape[1])*0.8].index, inplace = True)

    overlay_polygon_bridge_1 = gpd.overlay(layer_pl_candidate, layer_link_seeking, how='intersection', keep_geom_type=False)
    overlay_polygon_bridge_2 = gpd.overlay(layer_mask_description, layer_link_seeking, how='intersection', keep_geom_type=False)


    poly_id_to_drop = []
    seeking_id_to_drop = []
    # if multiple polygon candidates (map keys) are in the same link-seeking group, exclude them
    for sid in range(1, overlay_polygon_bridge_1.shape[0]+1):
        this_sid = overlay_polygon_bridge_1[(overlay_polygon_bridge_1['seeking_id']==sid)]

        if this_sid.shape[0] == layer_pl_candidate.shape[0]:
            seeking_id_to_drop.append(sid)
        elif this_sid.shape[0] > 1:
            for index_this_sid, row_this_sid in this_sid.iterrows():
                poly_id_to_drop.append(row_this_sid['pl_id'])
                seeking_id_to_drop.append(row_this_sid['seeking_id'])
            
    for targeted_poly_id in poly_id_to_drop:
        layer_pl_candidate.drop(layer_pl_candidate[layer_pl_candidate['pl_id'] == targeted_poly_id].index, inplace = True)
        overlay_polygon_bridge_1.drop(overlay_polygon_bridge_1[overlay_polygon_bridge_1['pl_id'] == targeted_poly_id].index, inplace = True)
    for targeted_seeking_id in seeking_id_to_drop:
        layer_link_seeking.drop(layer_link_seeking[layer_link_seeking['seeking_id'] == targeted_seeking_id].index, inplace = True)
        overlay_polygon_bridge_1.drop(overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id'] == targeted_seeking_id].index, inplace = True)
        overlay_polygon_bridge_2.drop(overlay_polygon_bridge_2[overlay_polygon_bridge_2['seeking_id'] == targeted_seeking_id].index, inplace = True)

    

    linked_poly_description = gpd.GeoDataFrame(columns=['pl_id', 'seeking_id', 'text_id1', 'text_id2', 'poly_geo', 'text_geo1', 'text_geo2', 'text_description1', 'text_description2', 'geometry'], crs=layer_mask_description.crs)
    for index, row in overlay_polygon_bridge_1.iterrows():
        this_poly_cand = overlay_polygon_bridge_1[(overlay_polygon_bridge_1['seeking_id'] == row['seeking_id'])]
        this_text = overlay_polygon_bridge_2[(overlay_polygon_bridge_2['seeking_id'] == row['seeking_id'])]

        for index_this_poly_cand, row_this_poly_cand in this_poly_cand.iterrows():
            this_poly = row_this_poly_cand
            break
        

        this_text = this_text.sort_values(by=['t_area'], ascending=False)
        poly_id = int(this_poly['pl_id'])
        poly_geo = this_poly['p_geometry']
        seeking_id = int(this_poly['seeking_id'])
        seeking_geo = this_poly['s_geometry']
        text_id_set = []
        text_geo_set = []
        text_description_set = []
        for index_this_text, row_this_text in this_text.iterrows():
            text_id_set.append(int(row_this_text['text_id']))
            text_geo_set.append(row_this_text['t_geometry'])
            text_description_set.append(row_this_text['text'])

        text_id_set = np.array(text_id_set)
        text_geo_set = np.array(text_geo_set)
        text_description_set = np.array(text_description_set)

        if this_text.shape[0] == 1:
            updated_record = gpd.GeoDataFrame([{'pl_id':poly_id, 'seeking_id':seeking_id, 'text_id1':text_id_set[0], 'text_id2':-1, 'poly_geo':str(poly_geo), 'text_geo1':text_geo_set[0].wkt, 'text_geo2':text_geo_set[0].wkt, 'text_description1':text_description_set[0], 'text_description2':'', 'geometry': overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id']==seeking_id].geometry.values[0]}])
        elif this_text.shape[0] > 1:
            updated_record = gpd.GeoDataFrame([{'pl_id':poly_id, 'seeking_id':seeking_id, 'text_id1':text_id_set[0], 'text_id2':text_id_set[1], 'poly_geo':str(poly_geo), 'text_geo1':text_geo_set[0].wkt, 'text_geo2':text_geo_set[1].wkt, 'text_description1':text_description_set[0], 'text_description2':text_description_set[1], 'geometry': overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id']==seeking_id].geometry.values[0]}])


        if this_text.shape[0] >= 1:
            linked_poly_description = gpd.GeoDataFrame(pd.concat( [linked_poly_description, updated_record], ignore_index=True), crs=layer_mask_description.crs)

            x1, y1, x2, y2 = overlay_polygon_bridge_1[overlay_polygon_bridge_1['seeking_id']==seeking_id].geometry.values[0].bounds
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(base_image2, (x1, y1), (x2, y2), (22,59,18), 4)
            if this_text.shape[0] == 1:
                cv2.putText(base_image2, str(text_description_set[0]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (22,59,18), 3)
            else:
                cv2.putText(base_image2, str(text_description_set[1] + ': ' + text_description_set[0]), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (22,59,18), 3)
        else:
            continue
    linked_poly_description.to_file(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_linked_pl_legend.geojson')), driver='GeoJSON')

    
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_linked_pl_legend.tif'))
    cv2.imwrite(out_file_path0, base_image2)

    return True





def integrating_result(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate7'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate7')))
    print('Step (8/9): Integrating results from previous steps...')


    map_name = target_map_name
    basemap_name = input_image
    base_image2 = cv2.imread(basemap_name)
    gray_mask = np.ones((base_image2.shape[0], base_image2.shape[1], base_image2.shape[2]), dtype='uint8')*50
    base_image2 = cv2.add(base_image2, gray_mask)

    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_legend.geojson')), driver='GeoJSON')
    linked_poly_description2 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_linked_pl_legend.geojson')), driver='GeoJSON')

    for index, row in linked_poly_description1.iterrows():
        x1, y1, x2, y2 = shapely.wkt.loads(row['poly_geo']).bounds
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(base_image2, (x1, y1), (x2, y2), (22,59,18), 4)
        if len(row['text_description2']) == 0:
            cv2.putText(base_image2, str(row['text_description1']), (x2+10, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (22,59,18), 3)
        else:
            cv2.putText(base_image2, str(row['text_description2'] + ': ' + row['text_description1']), (x2+20, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (22,59,18), 3)
    
    for index, row in linked_poly_description2.iterrows():
        x1, y1, x2, y2 = shapely.wkt.loads(row['poly_geo']).bounds
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(base_image2, (x1, y1), (x2, y2), (22,59,18), 4)
        cv2.putText(base_image2, str(row['text_description1']), (x2+10, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 1, (22,59,18), 3)

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_legend.tif'))
    cv2.imwrite(out_file_path0, base_image2)

    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_linked_pl_legend.tif')), os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_pl_legend.tif')))
    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_legend.geojson')), os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend.geojson')))
    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate6', map_name.replace('.tif', '_linked_pl_legend.geojson')), os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_pl_legend.geojson')))

    # ======

    map_name = target_map_name
    basemap_name = input_image
    base_image2 = cv2.imread(basemap_name)
    base_image3 = cv2.cvtColor(base_image2, cv2.COLOR_BGR2RGB)
    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate5', map_name.replace('.tif', '_linked_polygon_legend.geojson')), os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend.geojson')))

    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend.geojson')), driver='GeoJSON')
    #linked_poly_description2 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_pl_legend.geojson')), driver='GeoJSON')
    linked_poly_description1.drop('seeking_id', axis=1, inplace=True)
    linked_poly_description1.drop('text_id1', axis=1, inplace=True)
    linked_poly_description1.drop('text_id2', axis=1, inplace=True)
    linked_poly_description1.drop('poly_id', axis=1, inplace=True)
    linked_poly_description1['id'] = range(1, linked_poly_description1.shape[0]+1)
    linked_poly_description1.rename(columns={'text_description2': 'abbreviation', 'text_description1': 'name', 'text_geo2': 'abbr_geo', 'text_geo1': 'name_geo', }, inplace=True)
    linked_poly_description1['map_unit'] = None

    empty = []
    for iter in range(0, linked_poly_description1.shape[0]):
        empty.append('')
    defualt = []
    for iter in range(0, linked_poly_description1.shape[0]):
        defualt.append('solid') # solide, mixture
    linked_poly_description1['color'] = empty
    linked_poly_description1['pattern'] = defualt
    linked_poly_description1['description'] = empty
    linked_poly_description1['category'] = empty

    for index, row in linked_poly_description1.iterrows():
        candidate_description = row['name']
        if len(candidate_description.split('—')) > 1:
            if len(candidate_description.split('—')) > 2:
                linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'description'] = '—'.join(candidate_description.split('—')[1:])
            else:
                linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'description'] = candidate_description.split('—')[1:]
            linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'name'] = candidate_description.split('—')[0]
        

        xmin, ymin, xmax, ymax = shapely.wkt.loads(row['poly_geo']).bounds
        x_delta = xmax - xmin
        y_delta = ymax - ymin

        this_map_key = base_image3[int(ymin+y_delta*0.2): int(ymax-y_delta*0.2), int(xmin+x_delta*0.2): int(xmax-x_delta*0.2), :]

        black_threshold = 30
        white_threshold = 250

        rgb_trimmed = np.zeros((this_map_key.shape[2], this_map_key.shape[0], this_map_key.shape[1]), dtype='uint8')
        rgb_trimmed_source = np.copy(rgb_trimmed)
        rgb_trimmed = rgb_trimmed.astype(float)
        for dimension in range(0, 3):
            rgb_trimmed[dimension] = np.copy(this_map_key[:,:,dimension]).astype(float)

        rgb_trimmed_temp = np.copy(rgb_trimmed)
        rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan
        rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]<=black_threshold, rgb_trimmed_temp[1]<=black_threshold, rgb_trimmed_temp[2]<=black_threshold])] = np.nan

        rgb_trimmed[0, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        rgb_trimmed[1, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan
        rgb_trimmed[2, np.logical_and.reduce([rgb_trimmed_temp[0]>=white_threshold, rgb_trimmed_temp[1]>=white_threshold, rgb_trimmed_temp[2]>=white_threshold])] = np.nan

        if np.sum(np.isnan(rgb_trimmed)) >= (rgb_trimmed.shape[0]*rgb_trimmed.shape[1]*rgb_trimmed.shape[2]):
            median_color = np.array([int(np.nanquantile(rgb_trimmed_source[0],.5)),int(np.nanquantile(rgb_trimmed_source[1],.5)),int(np.nanquantile(rgb_trimmed_source[2],.5))])
        else:
            median_color = np.array([int(np.nanquantile(rgb_trimmed[0],.5)),int(np.nanquantile(rgb_trimmed[1],.5)),int(np.nanquantile(rgb_trimmed[2],.5))])
        
        this_hex = rgb2hex(median_color/255)
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'color'] = this_hex
    
    
    linked_poly_description1 = linked_poly_description1.set_crs('epsg:3857', allow_override=True)
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend_v2.geojson')), driver='GeoJSON')

    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '.tif'))
    cv2.imwrite(out_file_path0, base_image2)
    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend_v2.geojson')), driver='GeoJSON')


    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['name_geo']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    linked_poly_description1 = linked_poly_description1.set_crs('epsg:3857', allow_override=True)
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType_2.geojson')), driver='GeoJSON')

    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['poly_geo']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))

    linked_poly_description1.drop('poly_geo', axis=1, inplace=True)
    linked_poly_description1.drop('name_geo', axis=1, inplace=True)
    linked_poly_description1.drop('abbr_geo', axis=1, inplace=True)


    linked_poly_description1 = linked_poly_description1.set_crs('epsg:3857', allow_override=True)
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend_v3.geojson')), driver='GeoJSON')
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType.geojson')), driver='GeoJSON')


    # ======


    map_name = target_map_name
    basemap_name = input_image
    base_image2 = cv2.imread(basemap_name)
    base_image3 = cv2.cvtColor(base_image2, cv2.COLOR_BGR2RGB)

    #linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend.geojson')), driver='GeoJSON')
    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_pl_legend.geojson')), driver='GeoJSON')
    linked_poly_description1.drop('seeking_id', axis=1, inplace=True)
    linked_poly_description1.drop('text_id1', axis=1, inplace=True)
    linked_poly_description1.drop('text_id2', axis=1, inplace=True)
    linked_poly_description1.drop('text_description2', axis=1, inplace=True)
    linked_poly_description1.drop('text_geo2', axis=1, inplace=True)
    linked_poly_description1.drop('pl_id', axis=1, inplace=True)
    linked_poly_description1['id'] = range(1, linked_poly_description1.shape[0]+1)
    linked_poly_description1.rename(columns={'text_description1': 'name', 'text_geo1': 'name_geo'}, inplace=True)

    empty = []
    for iter in range(0, linked_poly_description1.shape[0]):
        empty.append('')
    linked_poly_description1['description'] = empty
    
    linked_poly_description1 = linked_poly_description1.set_crs('epsg:3857', allow_override=True)
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_pl_legend_v2.geojson')), driver='GeoJSON')
    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_pl_legend_v2.geojson')), driver='GeoJSON')

    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['name_geo']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    linked_poly_description1 = linked_poly_description1.set_crs('epsg:3857', allow_override=True)
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType_2.geojson')), driver='GeoJSON')

    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['poly_geo']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))

    linked_poly_description1.drop('poly_geo', axis=1, inplace=True)
    linked_poly_description1.drop('name_geo', axis=1, inplace=True)
    
    linked_poly_description1 = linked_poly_description1.set_crs('epsg:3857', allow_override=True)
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_pl_legend_v3.geojson')), driver='GeoJSON')
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType.geojson')), driver='GeoJSON')

    return True




def generating_geojson(target_map_name, input_image, path_to_intermediate):
    print('Step (9/9): Preparing output json - Generating JSON file...')


    map_name = target_map_name
    basemap_name = input_image

    legend_bounds_cand = []
    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_linked_polygon_legend_v2.geojson')), driver='GeoJSON')
    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['poly_geo']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    legend_bounds_cand.append(linked_poly_description1.total_bounds)
    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['name_geo']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    legend_bounds_cand.append(linked_poly_description1.total_bounds)
    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['abbr_geo']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    legend_bounds_cand.append(linked_poly_description1.total_bounds)
    legend_bounds_cand = np.array(legend_bounds_cand)

    this_legend_bounds = [legend_bounds_cand.min(axis=0)[0], legend_bounds_cand.min(axis=0)[1], legend_bounds_cand.max(axis=0)[2], legend_bounds_cand.max(axis=0)[3]]

    page_extraction_json = gpd.GeoDataFrame(columns=['name', 'ocr_text', 'color_estimation', 'bounds', 'model', 'geometry'], crs=linked_poly_description1.crs)
    updated_record = gpd.GeoDataFrame([{'name':'map_legend', 'ocr_text':'N.A.', 'color_estimation':None, 'bounds': str(geometry.box(*this_legend_bounds)), 'model':None, 'geometry': shapely.wkt.loads(str(geometry.box(*this_legend_bounds)))}])
    page_extraction_json = gpd.GeoDataFrame(pd.concat( [page_extraction_json, updated_record], ignore_index=True), crs=linked_poly_description1.crs)


    # ======


    in_path = os.path.join(path_to_intermediate, 'area_crop_binary.tif')
    out_path = os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_area_crop_binary.geojson'))

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

    map_area_detection = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_area_crop_binary.geojson')), driver='GeoJSON')


    map_area_detection['area'] = map_area_detection.geometry.area
    for index, poi in map_area_detection.iterrows():
        this_area_bounds = shapely.wkt.loads(str(poi['geometry']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
        break

    updated_record = gpd.GeoDataFrame([{'name':'map_content', 'ocr_text':'N.A.', 'color_estimation':None, 'bounds': str(this_area_bounds), 'model':None, 'geometry': this_area_bounds}])
    page_extraction_json = gpd.GeoDataFrame(pd.concat( [page_extraction_json, updated_record], ignore_index=True), crs=linked_poly_description1.crs)
    page_extraction_json = page_extraction_json.set_crs('epsg:3857', allow_override=True)
    page_extraction_json.to_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PageExtraction.geojson')), driver='GeoJSON')

    return True





def adjusting_crs(target_map_name, input_image, path_to_intermediate, output_dir, postprocessing_for_crs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    map_name = target_map_name
    basemap_name = input_image

    if postprocessing_for_crs == True:
        print('Step (9/9): Preparing output json - Adjusting CRS...')

        # convert the image to a binary raster .tif
        raster = rasterio.open(basemap_name)
        transform = raster.transform
        array     = raster.read(1)
        crs       = raster.crs 
        width     = raster.width 
        height    = raster.height 
        raster.close()

        this_epsg_code = pyproj.crs.CRS.from_proj4(crs.to_proj4()).to_epsg()
        #print('Target CRS:', this_epsg_code)
        trans_np = np.array(transform)
        trans_matrix = [trans_np[0], trans_np[1], trans_np[3], -trans_np[4], trans_np[2], trans_np[5]]
        #print(trans_matrix)

        original_file = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PageExtraction.geojson')), driver='GeoJSON')
        for index, poi in original_file.iterrows():
            geo_series = gpd.GeoSeries(poi['geometry'])
            original_file.loc[index, 'geometry'] = geo_series.affine_transform(trans_matrix).values[0]
        original_file = original_file.set_crs('epsg:'+str(this_epsg_code), allow_override=True)
        original_file.to_file(os.path.join(output_dir, map_name.replace('.tif', '_PageExtraction.geojson')), driver='GeoJSON')


        original_file = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType.geojson')), driver='GeoJSON')
        for index, poi in original_file.iterrows():
            geo_series = gpd.GeoSeries(poi['geometry'])
            original_file.loc[index, 'geometry'] = geo_series.affine_transform(trans_matrix).values[0]
        original_file = original_file.set_crs('epsg:'+str(this_epsg_code), allow_override=True)
        original_file.to_file(os.path.join(output_dir, map_name.replace('.tif', '_PolygonType.geojson')), driver='GeoJSON')


        original_file = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType.geojson')), driver='GeoJSON')
        for index, poi in original_file.iterrows():
            geo_series = gpd.GeoSeries(poi['geometry'])
            original_file.loc[index, 'geometry'] = geo_series.affine_transform(trans_matrix).values[0]
        original_file = original_file.set_crs('epsg:'+str(this_epsg_code), allow_override=True)
        original_file.to_file(os.path.join(output_dir, map_name.replace('.tif', '_PointLineType.geojson')), driver='GeoJSON')
    else:
        shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PageExtraction.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PageExtraction.geojson')))
        shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PolygonType.geojson')))
        shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PointLineType.geojson')))

    print('Output at ' + str(output_dir)+' as json files...')
    print(os.path.join(output_dir, map_name.replace('.tif', '_PageExtraction.geojson')), 'for map area segmentation')
    print(os.path.join(output_dir, map_name.replace('.tif', '_PolygonType.geojson')), 'for legend item segmentation (polygon)')
    print(os.path.join(output_dir, map_name.replace('.tif', '_PointLineType.geojson')), 'for legend item segmentation (point, line)')

    return True



def start_linking(target_map_name, input_image, output_dir, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, preprocessing_for_cropping, postprocessing_for_crs):
    map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping)
    text_spotting_with_pytesseract(target_map_name, input_image, path_to_intermediate)
    read_results_from_pytesseract(target_map_name, input_image, path_to_intermediate)
    read_results_from_mapkurator(target_map_name, input_image, path_to_intermediate, path_to_mapkurator_output)
    map_key_extraction_polygon(target_map_name, input_image, path_to_intermediate)
    linking_description_polygon(target_map_name, input_image, path_to_intermediate)
    searching_possible_description(target_map_name, input_image, path_to_intermediate)
    linking_description_pointline(target_map_name, input_image, path_to_intermediate)
    integrating_result(target_map_name, input_image, path_to_intermediate)
    generating_geojson(target_map_name, input_image, path_to_intermediate)
    adjusting_crs(target_map_name, input_image, path_to_intermediate, output_dir, postprocessing_for_crs)


