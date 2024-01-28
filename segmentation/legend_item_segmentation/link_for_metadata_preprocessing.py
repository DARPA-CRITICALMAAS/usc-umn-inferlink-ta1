
import numpy as np
from matplotlib.colors import rgb2hex

import os
import cv2

import pytesseract
tesseract_exe = os.getenv("TESSERACT_EXE", r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe')
pytesseract.pytesseract.tesseract_cmd = tesseract_exe

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

import json



def processing_uncharted_json_batch(input_legend_segmentation, target_map_name, output_segmentation):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    target_id = -1
    for this_gj in gj['images']:
        this_id = this_gj['id']
        this_file_name = this_gj['file_name']
        #print(target_map_name.split('.')[0], this_file_name)
        if target_map_name.split('.')[0] in this_file_name.replace('%20', ' '):
            target_id = this_id
            legend_area_placeholder = np.zeros((this_gj['height'], this_gj['width']), dtype='uint8')
            break
    
    if target_id == -1:
        return False
    
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


def processing_uncharted_json_single(input_image, input_legend_segmentation, target_map_name, output_segmentation):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    img0 = cv2.imread(input_image)
    gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    legend_area_placeholder = np.zeros((gray0.shape[0],gray0.shape[1]), dtype='uint8')

    for this_gj in gj['segments']:
        if 'legend_polygons' in this_gj['class_label'] or 'legend_points_lines' in this_gj['class_label']:
            cv2.fillConvexPoly(legend_area_placeholder, np.array(this_gj['poly_bounds']), 1)
            legend_area_placeholder[legend_area_placeholder > 0] = 255
    
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), legend_area_placeholder)
    return True
                    

def map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, competition_custom):
    print('Legend-item Segmentation is working on input image:', input_image)
    print('')

    if not os.path.exists(path_to_intermediate):
            os.makedirs(path_to_intermediate)

    output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')

    flag_identify = False

    if len(input_legend_segmentation) > 0:
        # if have legend segmentation result (a binary mask that highlights polygon/line/point legends)...
        print('Step ( 0/10): Processing legend-area segmentation result...')
        #print('Step (0/9): Processing the given map legend segmentation (whole area) result from... '+str(input_legend_segmentation))
        #print('*Please note that the output json for map area segmentation has not directly used this source...')
        if '.tif' not in input_legend_segmentation:
            #print('    Input for legend_area segmentation is not given as a single tif file; will process the json file first...')
            if competition_custom == False:
                try:
                    flag_identify = processing_uncharted_json_single(input_image, input_legend_segmentation, target_map_name, output_segmentation)
                except:
                    flag_identify = processing_uncharted_json_batch(input_legend_segmentation, target_map_name, output_segmentation)
            elif competition_custom == True:
                try:
                    flag_identify = processing_uncharted_json_batch(input_legend_segmentation, target_map_name, output_segmentation)
                except:
                    flag_identify = processing_uncharted_json_single(input_image, input_legend_segmentation, target_map_name, output_segmentation)

            if flag_identify == False:
                return False
        else:
            shutil.copyfile(input_legend_segmentation, output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'))
        solution = cv2.imread(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'))
        solution = cv2.cvtColor(solution, cv2.COLOR_BGR2GRAY)
    else:
        print('Require legend-area segmentation output from Uncharted......')

        return False
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
    print('Step ( 1/10): Text spotting with pytesseract...')


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
    print('Step ( 2/10): Processing results from pytesseract...')


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

    '''
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
    '''
    

    return True





def read_results_from_mapkurator(target_map_name, input_image, path_to_intermediate, path_to_mapkurator_output):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate3_2'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate3_2')))
    print('Step ( 3/10): Processing results from mapkurator...')


    map_name = target_map_name
    #mapkurator_source_name = os.path.join(path_to_mapkurator_output, map_name.replace('.tif', '.geojson'))
    if os.path.isfile(path_to_mapkurator_output) == True:
        mapkurator_source_name = path_to_mapkurator_output
        mapkurator_name = os.path.join(path_to_intermediate, 'intermediate3_2', map_name.replace('.tif', '_v2.geojson'))

        with open(mapkurator_source_name, 'r') as file:
            source_data = file.read().replace('-', '')
        with open(mapkurator_name, 'w') as file:
            file.write(source_data)
    else:
        mapkurator_name = os.path.join(path_to_intermediate, 'intermediate9', map_name.replace('.tif', '_tesseract_v2.geojson'))
    
    
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')
    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

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


    '''
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
    '''

    return True






def map_key_extraction_polygon(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate4'))):
        os.makedirs(os.path.join(path_to_intermediate,  str('intermediate4')))
    print('Step ( 4/10): Applying metadata preprocessing for legend-item features...')


    map_name = target_map_name
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')

    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    basemap_gray_name = os.path.join(path_to_intermediate, 'area_crop_binary.tif')
    base_gray_image = cv2.imread(basemap_gray_name)
    base_gray_image = cv2.cvtColor(base_gray_image, cv2.COLOR_BGR2GRAY)

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
    base_image_mask_black = cv2.bitwise_and(base_image_mask_black, base_gray_image)
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
    
    '''
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
    '''

    
    '''
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
    '''

    '''
    
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
    '''

    return True










def start_linking(target_map_name, input_image, output_dir, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, preprocessing_for_cropping, postprocessing_for_crs, competition_custom):
    flag_identify = map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, competition_custom)
    if flag_identify == True:
        text_spotting_with_pytesseract(target_map_name, input_image, path_to_intermediate)
        read_results_from_pytesseract(target_map_name, input_image, path_to_intermediate)
        read_results_from_mapkurator(target_map_name, input_image, path_to_intermediate, path_to_mapkurator_output)
        map_key_extraction_polygon(target_map_name, input_image, path_to_intermediate)
    else:
        with open('missing.csv','a') as fd:
            fd.write(str(target_map_name)+'\n')
            fd.close()


def main():
    #flag_skipping = False
    for target_map_cand in os.listdir('Data/validation/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]
            print('Processing map... '+str(target_map)+'.tif')

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/validation/'+str(target_map)+'.tif'
            path_to_intermediate = 'LINK_Intermediate/validation/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-validation/'+str(target_map)+'.geojson'

            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True)

    for target_map_cand in os.listdir('Data/testing/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/testing/'+str(target_map)+'.tif'
            path_to_intermediate = 'LINK_Intermediate/testing/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-evaluation/'+str(target_map)+'.geojson'

            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True)

    for target_map_cand in os.listdir('Data/training/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/training/'+str(target_map)+'.tif'
            path_to_intermediate = 'LINK_Intermediate/training/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_training_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-training/'+str(target_map)+'.geojson'

            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True)


    

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()

