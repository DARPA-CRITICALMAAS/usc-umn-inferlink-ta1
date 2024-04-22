
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

from datetime import datetime



def processing_uncharted_json_batch(input_legend_segmentation, target_map_name, output_segmentation, placeholder_handler):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    target_id = -1
    for this_gj in gj['images']:
        this_id = this_gj['id']
        this_file_name = this_gj['file_name']
        #print(target_map_name.split('.')[0], this_file_name)
        if target_map_name.split('.')[0] in this_file_name.replace('%20', ' '):
            target_id = this_id
            legend_area_placeholder = np.zeros((this_gj['height'], this_gj['width']), dtype=np.int32)
            legend_area_placeholder0 = np.zeros((this_gj['height'], this_gj['width']), dtype=np.int32)
            legend_area_placeholder1 = np.zeros((this_gj['height'], this_gj['width']), dtype=np.int32)
            legend_area_placeholder2 = np.zeros((this_gj['height'], this_gj['width']), dtype=np.int32)
            break
    
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), legend_area_placeholder.astype(np.int8))
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_poly.tif'), legend_area_placeholder0.astype(np.int8))
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif'), legend_area_placeholder1.astype(np.int8))
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_roi.tif'), legend_area_placeholder2.astype(np.int8))
    if target_id == -1:
        return False, False
    
    segmentation_added = 0
    for this_gj in gj['annotations']:
        if this_gj['image_id'] == target_id:
            if this_gj['category_id'] == 1:
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

                cv2.fillConvexPoly(legend_area_placeholder0, poly_coord, 1)
                legend_area_placeholder0[legend_area_placeholder0 > 0] = 255
                cv2.fillConvexPoly(legend_area_placeholder, poly_coord, 1)
                legend_area_placeholder[legend_area_placeholder > 0] = 255
                cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_poly.tif'), legend_area_placeholder0)

                segmentation_added += 1
            
            if this_gj['category_id'] == 0:
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

                cv2.fillConvexPoly(legend_area_placeholder1, poly_coord, 1)
                legend_area_placeholder1[legend_area_placeholder1 > 0] = 255
                cv2.fillConvexPoly(legend_area_placeholder, poly_coord, 1)
                legend_area_placeholder[legend_area_placeholder > 0] = 255
                cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif'), legend_area_placeholder1)

                segmentation_added += 1
            

            if this_gj['category_id'] == 2:
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

                cv2.fillConvexPoly(legend_area_placeholder2, poly_coord, 1)
                legend_area_placeholder2[legend_area_placeholder2 > 0] = 255
                cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_roi.tif'), legend_area_placeholder2)

                segmentation_added += 1
            
            if segmentation_added >= 3:
                break
    
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), legend_area_placeholder)

    placeholder_generated = False
    if placeholder_handler == True and np.unique(legend_area_placeholder).shape[0] == 1:
        print('------------- No legend area was found, generate a placeholder for possible point and line legend items...')
        legend_area_placeholder1[max(0, legend_area_placeholder1.shape[0]-60):, :] = 255
        cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif'), legend_area_placeholder1)
        cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), legend_area_placeholder1)
        placeholder_generated = True
    return True, placeholder_generated



def processing_uncharted_json_single(input_image, input_legend_segmentation, target_map_name, output_segmentation, placeholder_handler):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    img0 = cv2.imread(input_image)
    gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    legend_area_placeholder = np.zeros((gray0.shape[0],gray0.shape[1]), dtype=np.int32)
    legend_area_placeholder0 = np.zeros((gray0.shape[0],gray0.shape[1]), dtype=np.int32)
    legend_area_placeholder1 = np.zeros((gray0.shape[0],gray0.shape[1]), dtype=np.int32)
    legend_area_placeholder2 = np.zeros((gray0.shape[0],gray0.shape[1]), dtype=np.int32)

    for this_gj in gj['segments']:
        if 'legend_polygons' in this_gj['class_label']:
            cv2.fillConvexPoly(legend_area_placeholder0, np.array(this_gj['poly_bounds'], dtype=np.int32), 1)
            legend_area_placeholder0[legend_area_placeholder0 > 0] = 255
            cv2.fillConvexPoly(legend_area_placeholder, np.array(this_gj['poly_bounds'], dtype=np.int32), 1)
            legend_area_placeholder[legend_area_placeholder > 0] = 255
        if 'legend_points_lines' in this_gj['class_label']:
            cv2.fillConvexPoly(legend_area_placeholder1, np.array(this_gj['poly_bounds'], dtype=np.int32), 1)
            legend_area_placeholder1[legend_area_placeholder1 > 0] = 255
            cv2.fillConvexPoly(legend_area_placeholder, np.array(this_gj['poly_bounds'], dtype=np.int32), 1)
            legend_area_placeholder[legend_area_placeholder > 0] = 255
        if 'map' in this_gj['class_label']:
            cv2.fillConvexPoly(legend_area_placeholder2, np.array(this_gj['poly_bounds'], dtype=np.int32), 1)
            legend_area_placeholder2[legend_area_placeholder2 > 0] = 255

    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_poly.tif'), legend_area_placeholder0.astype(np.int8))
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif'), legend_area_placeholder1.astype(np.int8))
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_roi.tif'), legend_area_placeholder2.astype(np.int8))
    cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), legend_area_placeholder.astype(np.int8))

    placeholder_generated = False
    if placeholder_handler == True and np.unique(legend_area_placeholder).shape[0] == 1 and np.mean(legend_area_placeholder) == 0:
        print('------------- No legend area was found, generate a placeholder for possible point and line legend items...')
        legend_area_placeholder1[max(0, legend_area_placeholder1.shape[0]-60):, :] = 255
        cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif'), legend_area_placeholder1)
        cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'), legend_area_placeholder1)
        placeholder_generated = True
    return True, placeholder_generated
                    

def map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, placeholder_handler, competition_custom):
    print('\n\n\n\n\n')
    print('------------- Legend-item Segmentation is working on input image:', input_image)
    print('')

    if not os.path.exists(path_to_intermediate):
            os.makedirs(path_to_intermediate)

    output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')

    flag_identify = False
    placeholder_generated = False

    if len(input_legend_segmentation) > 0:
        # if have legend segmentation result (a binary mask that highlights polygon/line/point legends)...
        print('Step ( 1/10): Processing legend-area segmentation result...')
        #print('Step (0/9): Processing the given map legend segmentation (whole area) result from... '+str(input_legend_segmentation))
        #print('*Please note that the output json for map area segmentation has not directly used this source...')
        if '.tif' not in input_legend_segmentation:
            #print('    Input for legend_area segmentation is not given as a single tif file; will process the json file first...')
            if competition_custom == False:
                try:
                    flag_identify, placeholder_generated = processing_uncharted_json_single(input_image, input_legend_segmentation, target_map_name, output_segmentation, placeholder_handler)
                except:
                    flag_identify, placeholder_generated = processing_uncharted_json_batch(input_legend_segmentation, target_map_name, output_segmentation, placeholder_handler)
            elif competition_custom == True:
                try:
                    flag_identify, placeholder_generated = processing_uncharted_json_batch(input_legend_segmentation, target_map_name, output_segmentation, placeholder_handler)
                except:
                    flag_identify, placeholder_generated = processing_uncharted_json_single(input_image, input_legend_segmentation, target_map_name, output_segmentation, placeholder_handler)

            if flag_identify == False:
                return False, None
        else:
            shutil.copyfile(input_legend_segmentation, output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'))
        solution = cv2.imread(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif'))
        try:
            solution = cv2.cvtColor(solution, cv2.COLOR_BGR2GRAY)
        except:
            print('------------- Unable to proceed with cv2.COLOR_BGR2GRAY...')
            return False, None
    else:
        print('------------- Require legend-area segmentation output from Uncharted......')

        return False, None
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

    return True, placeholder_generated






def text_spotting_with_pytesseract(target_map_name, input_image, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, 'intermediate9')):
        os.makedirs(os.path.join(path_to_intermediate, 'intermediate9'))

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








def read_results_from_mapkurator(target_map_name, path_to_intermediate, path_to_mapkurator_output):
    


    map_name = target_map_name
    #mapkurator_source_name = os.path.join(path_to_mapkurator_output, map_name.replace('.tif', '.geojson'))
    if os.path.isfile(path_to_mapkurator_output) == True:
        mapkurator_source_name = path_to_mapkurator_output
        mapkurator_name = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_v2.geojson'))

        with open(mapkurator_source_name, 'r') as file:
            source_data = file.read().replace('-', '')
        with open(mapkurator_name, 'w') as file:
            file.write(source_data)
    else:
        print('------------- Results from MapKurator are not available or accessible, please check the file if you intend to include this input...')
        print('------------- Will be processing pyTesseract instead...')
        text_spotting_with_pytesseract(target_map_name, None, path_to_intermediate)
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
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_mapkurator_mask.tif'))
    cv2.imwrite(out_file_path0, text_mask)


    text_mask_buffer = np.copy(text_mask)
    kernel = np.ones((2, 1), np.uint8)
    text_mask_buffer = cv2.erode(text_mask_buffer, kernel, iterations = 1)
    kernel = np.ones((1, 5), np.uint8)
    text_mask_buffer = cv2.dilate(text_mask_buffer, kernel, iterations = 1)
    for i in range(30):
        text_mask_buffer[:, 1:] = np.maximum(text_mask_buffer[:, 1:], text_mask_buffer[:, 0:-1]) # growing right
    cv2.imwrite(os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_mapkurator_mask_buffer_v1.tif')), text_mask_buffer)

    '''
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
    out_file_path0 = os.path.join(path_to_intermediate, 'intermediate3', map_name.replace('.tif', '_mapkurator_mask_buffer_v2.tif'))
    cv2.imwrite(out_file_path0, text_mask_buffer)
    '''


    return True




def rectangularization(path_to_intermediate, intermediate_stage, target_map_name, targeted_block):
    ### rectangularization
    # not deployed yet, can be modified for both poly and ptln
    # tif > geojson
    base_image = cv2.imread(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_candidate_ptln2.tif')))
    input_raster = os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_candidate_ptln2.png'))
    output_vector = os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_candidate_ptln2.geojson'))
    cv2.imwrite(input_raster, base_image)

    in_path = input_raster
    out_path = output_vector

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

    polygon_extraction = gpd.read_file(output_vector, driver='GeoJSON')
    mirrored_polygon = gpd.GeoDataFrame(columns=['id', 'geometry'], crs=polygon_extraction.crs)
    for index, poi in polygon_extraction.iterrows():
        if index == polygon_extraction.shape[0]-1:
            break

        # convert to bounding box
        bbox_tuple = poi['geometry'].bounds
        geom = shapely.geometry.box(*bbox_tuple)
        #print(this_mirrored_polygon)
        #print(bbox_tuple, ' >>> ', geom)
        this_mirrored_polygon = geom

        updated_record = gpd.GeoDataFrame([{'id': index, 'geometry':this_mirrored_polygon}])
        mirrored_polygon = gpd.GeoDataFrame(pd.concat( [mirrored_polygon, updated_record], ignore_index=True), crs=polygon_extraction.crs)

    mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
    mirrored_polygon.to_file(output_vector.replace('_candidate_ptln2.geojson', '_candidate_ptln3.geojson'), driver='GeoJSON')


    # geojson > tif
    source_name = output_vector.replace('_candidate_ptln2.geojson', '_candidate_ptln3.geojson')
    basemap_name = os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_candidate_ptln2.tif'))

    base_image = cv2.imread(basemap_name)
    base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

    gdf = gpd.read_file(source_name, driver='GeoJSON')
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
        if poi['area'] > (base_image.shape[0] * base_image.shape[1]) * 0.8:
            continue
        this_text_mask = rasterio.features.rasterize([gdf.loc[index]['geometry']], out_shape=(base_image.shape[0], base_image.shape[1]))
        text_mask = cv2.bitwise_or(text_mask, this_text_mask)
        
    text_mask[text_mask > 0] = 255

    kernel = np.ones((3, 3), np.uint8)
    ptln_candidate_rectangularized = cv2.dilate(text_mask, kernel, iterations = 1)
    ptln_candidate_rectangularized = cv2.bitwise_and(targeted_block, ptln_candidate_rectangularized)
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_candidate_ptln3.tif')), ptln_candidate_rectangularized)





def refine_poly_roi(target_map_name, path_to_intermediate):
    output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')
    segmentatation_poly = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_poly.tif')
    roi_poly = cv2.imread(segmentatation_poly)
    roi_poly = cv2.cvtColor(roi_poly, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(roi_poly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_contour_size = 0
    valid_contours = None
    # list for storing names of shapes
    for c in contours:
        # Discard shapes that are too large or small to be a valid legend
        area = cv2.contourArea(c)
        if area > max_contour_size and area < roi_poly.shape[0]*roi_poly.shape[1]:
            max_contour_size = area
            valid_contours = c
    

    
    print('Step ( 4/10): Tranferring from contours to polygons...')
    roi_img = np.ones((roi_poly.shape[0], roi_poly.shape[1]), dtype='uint8')
    if valid_contours is not None:
        cv2.drawContours(roi_img, [valid_contours], 0, 255, 1)

    # flood fill background to find inner holes
    floodfill_candidate = np.ones((roi_img.shape[0], roi_img.shape[1]), dtype='uint8') * 255
    holes = np.copy(roi_img)
    cv2.floodFill(holes, None, (0, 0), 255)

    # invert holes mask, bitwise or with img fill in holes
    holes = cv2.bitwise_not(holes)
    valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
    floodfill_block = cv2.bitwise_and(255-roi_img, valid_holes)
    floodfill_block[floodfill_block > 0] = 255 # 254

    shutil.copyfile(os.path.join(path_to_intermediate, 'area_crop_poly.tif'), os.path.join(path_to_intermediate, 'area_crop_poly_v0.tif'))
    cv2.imwrite(os.path.join(path_to_intermediate, 'area_crop_poly.tif'), floodfill_block)




    segmentatation_poly = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif')
    roi_ptln = cv2.imread(segmentatation_poly)
    roi_ptln = cv2.cvtColor(roi_ptln, cv2.COLOR_BGR2GRAY)

    #segmentatation_poly = output_segmentation
    #roi = cv2.imread(segmentatation_poly)
    #roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.bitwise_or(floodfill_block, roi_ptln)
    shutil.copyfile(os.path.join(path_to_intermediate, 'area_crop_binary.tif'), os.path.join(path_to_intermediate, 'area_crop_binary_v0.tif'))
    cv2.imwrite(os.path.join(path_to_intermediate, 'area_crop_binary.tif'), roi)


    segmentatation_poly = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_rgb.tif')
    roi_rgb = cv2.imread(segmentatation_poly)
    roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.bitwise_and(roi_rgb, roi_rgb, mask=roi)
    roi_rgb = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    shutil.copyfile(os.path.join(path_to_intermediate, 'area_crop_rgb.tif'), os.path.join(path_to_intermediate, 'area_crop_rgb_v0.tif'))
    cv2.imwrite(os.path.join(path_to_intermediate, 'area_crop_rgb.tif'), roi_rgb)




    return







def map_key_extraction(target_map_name, path_to_intermediate, path_to_mapkurator_output, placeholder_generated, only_poly):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate2'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate2')))
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate3'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate3')))
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate4'))):
        os.makedirs(os.path.join(path_to_intermediate,  str('intermediate4')))



    refine_poly_roi(target_map_name, path_to_intermediate)


    print('Step ( 2/10): Applying color thresholding and contour finding...')

    output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')
    segmentatation_poly = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_poly.tif')
    roi_poly = cv2.imread(segmentatation_poly)
    roi_poly = cv2.cvtColor(roi_poly, cv2.COLOR_BGR2GRAY)

    segmentatation_ptln = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif')
    roi_ptln = cv2.imread(segmentatation_ptln)
    roi_ptln = cv2.cvtColor(roi_ptln, cv2.COLOR_BGR2GRAY)
    
    basemap_name = os.path.join(path_to_intermediate, 'area_crop_rgb.tif')
    source_basemap = cv2.imread(basemap_name)
    img_rgb = cv2.cvtColor(source_basemap, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(source_basemap, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(source_basemap, cv2.COLOR_BGR2GRAY)
    
    '''
    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_canny_0.tif')), canny)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_canny_1.tif')), canny_dilate)
    '''

    # setting threshold of gray image
    _, threshold = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) # 60, 255

    # using a findContours() function
    kernel = np.ones((2, 4), np.uint8)
    threshold_dilate = cv2.dilate(255-threshold, kernel, iterations = 1)
    #threshold_dilate = cv2.erode(threshold_dilate, kernel, iterations = 1)
    threshold_dilate = 255 - threshold_dilate

    _, threshold_dilate = cv2.threshold(threshold_dilate, 127, 255, cv2.THRESH_BINARY) # 60, 255
    contours, hierarchy = cv2.findContours(threshold_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    adaptive_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)



    ### Use a more strict threshold to handle polygon legend items with mixture contents
    # setting threshold of gray image
    _, threshold_v2 = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY) # 60, 255

    # using a findContours() function
    kernel = np.ones((2, 4), np.uint8)
    threshold_dilate_v2 = cv2.dilate(255-threshold_v2, kernel, iterations = 1)
    #threshold_dilate_v2 = cv2.erode(threshold_dilate_v2, kernel, iterations = 1)
    threshold_dilate_v2 = 255 - threshold_dilate_v2

    _, threshold_dilate_v2 = cv2.threshold(threshold_dilate_v2, 60, 255, cv2.THRESH_BINARY) # 60, 255
    contours_v2, hierarchy_v2 = cv2.findContours(threshold_dilate_v2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, hierarchy = cv2.findContours(threshold_v2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    adaptive_threshold_v2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)


    #threshold = cv2.bitwise_and(threshold, threshold_v2)
    #threshold_dilate = cv2.bitwise_and(threshold_dilate, threshold_dilate_v2)
    #adaptive_threshold = cv2.bitwise_or(adaptive_threshold, adaptive_threshold_v2)

    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_thresholding_fixed_0_v1.tif')), threshold)
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_thresholding_fixed_1_v1.tif')), threshold_dilate)
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_thresholding_adaptive_v1.tif')), adaptive_threshold)

    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_thresholding_fixed_0_v2.tif')), threshold_v2)
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_thresholding_fixed_1_v2.tif')), threshold_dilate_v2)
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_thresholding_adaptive_v2.tif')), adaptive_threshold_v2)


    print('Step ( 3/10): Filtering contours for poly legend items...')

    # Reducing selection to contours that are quaderlaterals of reasonable size.
    max_contour_size = 50000 # 1% of image
    min_contour_size = 1000

    candidate_areas = []
    candidate_rect_areas = []
    candidate_contours = []
    candidate_width = []
    i = 0
    # list for storing names of shapes
    for c in contours:
        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)

        # Only quadrilaterals shapes are valid
        if len(approx) != 4: # or len(approx) != 3:
            continue

        # Discard shapes that are too large or small to be a valid legend
        area = cv2.contourArea(c)
        if area <= min_contour_size or area >= max_contour_size:
            continue

        # Discard shapes that are quads but not rectangular
        x,y,w,h = cv2.boundingRect(c)
        rect_area = w*h
        if rect_area > area*4:
            continue

        candidate_areas.append(area)
        candidate_rect_areas.append(rect_area)
        candidate_contours.append(c)
        candidate_width.append(w)
    

    ##################
    i = 0
    # list for storing names of shapes
    for c in contours_v2:
        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)

        # Only quadrilaterals shapes are valid
        if len(approx) != 4: # or len(approx) != 3:
            continue

        # Discard shapes that are too large or small to be a valid legend
        area = cv2.contourArea(c)
        if area <= min_contour_size or area >= max_contour_size:
            continue

        # Discard shapes that are quads but not rectangular
        x,y,w,h = cv2.boundingRect(c)
        rect_area = w*h
        if rect_area > area*4:
            continue

        candidate_areas.append(area)
        candidate_rect_areas.append(rect_area)
        candidate_contours.append(c)
        candidate_width.append(w)
    ##################

    # Calculate valid area threshold
    valid_bounds = 0.66 # Percent +/- threshold from median
    candidate_df = pd.DataFrame({'area' : candidate_areas, 'rect_area' : candidate_rect_areas, 'contour' :candidate_contours})
    candidate_areas.sort()
    candidate_rect_areas.sort()

    median_area = np.median(candidate_areas)
    min_valid_area = median_area - median_area*valid_bounds
    max_valid_area = median_area + median_area*valid_bounds

    median_rect_area = np.median(candidate_rect_areas)
    min_valid_rect_area = median_rect_area - median_rect_area*valid_bounds
    max_valid_rect_area = median_rect_area + median_rect_area*valid_bounds

    candidate_width = np.array(candidate_width)
    median_width = np.median(candidate_width)

    '''
    # Plot Bar graphs of area
    plt.rcParams['figure.dpi'] = 100
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # Contour Area
    ax[0].set_title('Plot of candidate contours area')
    ax[0].set_ylabel('Area of contour')
    ax[0].bar(range(0,len(candidate_areas)), candidate_areas)
    ax[0].axhline(y=max_valid_area, color = 'r', linestyle = '-', label='valid area range')
    ax[0].axhline(y=min_valid_area, color = 'r', linestyle = '-')
    ax[0].legend(loc='upper left')

    # Bounding Box Area
    ax[1].set_title('Plot of contours bounding box area')
    ax[1].set_ylabel('Area of contour')
    ax[1].bar(range(0,len(candidate_rect_areas)), candidate_rect_areas)
    ax[1].axhline(y=min_valid_rect_area, color = 'r', linestyle = '-', label='valid area range')
    ax[1].axhline(y=max_valid_rect_area, color = 'r', linestyle = '-')
    ax[1].legend(loc='upper left')

    fig.show()
    '''

    valid_contours = []
    for c in candidate_contours:
        # Discard shapes outside the valid contour areas
        area = cv2.contourArea(c)
        if area <= min_valid_rect_area or area >= max_valid_rect_area:
            continue

        valid_contours.append(c)


    
    print('Step ( 4/10): Tranferring from contours to polygons...')

    poly_img = np.ones((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), dtype='uint8')

    # list for storing names of shapes
    for i in range(0,len(valid_contours)):
        # using drawContours() function
        cv2.drawContours(poly_img, [valid_contours[i]], 0, (0, 0, 255), 1)
    poly_img = cv2.cvtColor(poly_img, cv2.COLOR_RGB2GRAY)

    # flood fill background to find inner holes
    floodfill_candidate = np.ones((poly_img.shape[0], poly_img.shape[1]), dtype='uint8') * 255
    holes = np.copy(poly_img)
    cv2.floodFill(holes, None, (0, 0), 255)

    # invert holes mask, bitwise or with img fill in holes
    holes = cv2.bitwise_not(holes)
    valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
    floodfill_block = cv2.bitwise_and(255-poly_img, valid_holes)
    #print(np.unique(floodfill_block))
    floodfill_block[floodfill_block > 0] = 255 # 254
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly2.tif')), floodfill_block)
    shutil.copyfile(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly2.tif')), 
                    os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly_source.tif')))
    poly_candidate_sweeping = np.copy(floodfill_block)

    kernel = np.ones((5, 5), np.uint8)
    floodfill_block_erosion = cv2.erode(floodfill_block, kernel, iterations = 1)



    adaptive_processing = False
    kernel = np.ones((5, 10), np.uint8)
    floodfill_block_erosion_test = cv2.erode(floodfill_block, kernel, iterations = 1)
    print('------------- '+str(np.mean(floodfill_block_erosion_test) / max(1, np.max(floodfill_block_erosion_test))))
    if np.unique(floodfill_block_erosion_test).shape[0] < 2:
        # if there are 'almost' no pixels corresponding to polygon legend items...
        adaptive_processing = True   

    #if adaptive_processing == False:
        #print('early stop...')
        #return     


    if adaptive_processing == True and placeholder_generated == False:
        print('------------- Fragmented items; will proceed with an additional segmentation...')
        print('------------- This fragmented output might due to a fuzzy image - low resolutions or scanning issues...')
        output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')
        segmentatation_all = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_binary.tif')

        #print(output_segmentation)
        roi_all = cv2.imread(segmentatation_all)
        roi_all = cv2.cvtColor(roi_all, cv2.COLOR_BGR2GRAY)

        white_place_holder = np.ones((roi_all.shape[0], roi_all.shape[1]), dtype='uint8')*255
        enhanced_img_gray = cv2.imread(os.path.join(path_to_intermediate, 'area_crop_rgb.tif'))
        enhanced_img_gray = cv2.cvtColor(enhanced_img_gray, cv2.COLOR_BGR2GRAY)
        enhanced_img_gray = cv2.copyTo(src=enhanced_img_gray, dst=white_place_holder, mask=roi_all)
        blur_img = cv2.medianBlur(enhanced_img_gray, 5) 
        adaptive = cv2.adaptiveThreshold(blur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_adaptive2.tif')), adaptive)

        adaptive = 255 - adaptive
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_adaptive3.tif')), adaptive)

        kernel = np.ones((5, 10), np.uint8)
        adaptive_buffer = cv2.dilate(adaptive, kernel, iterations = 1)

        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        #opening = cv2.morphologyEx(adaptive_buffer, cv2.MORPH_OPEN, kernel, iterations=1)
        #adaptive_buffer = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY)[1]

        # flood fill background to find inner holes
        floodfill_candidate = np.ones((adaptive_buffer.shape[0], adaptive_buffer.shape[1]), dtype='uint8') * 255
        holes = np.copy(adaptive_buffer)
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
        floodfill_block = cv2.bitwise_or(adaptive_buffer, valid_holes)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_floodfill.tif')), floodfill_block)

        floodfill_block_backup = np.copy(floodfill_block)

        # mainly based on floodfill
        floodfill_block = cv2.bitwise_and(floodfill_block, 255-adaptive_buffer)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_floodfill2.tif')), floodfill_block)

        floodfill_candidate = np.ones((floodfill_block.shape[0], floodfill_block.shape[1]), dtype='uint8') * 255
        holes = np.copy(floodfill_block)
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
        floodfill_block = cv2.bitwise_or(floodfill_block, valid_holes)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_floodfill3.tif')), floodfill_block)

        
        kernel = np.ones((15, 15), np.uint8) # (10, 10)
        floodfill_block_erosion = cv2.erode(floodfill_block, kernel, iterations = 1)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_floodfill4.tif')), floodfill_block_erosion)




    print('Step ( 5/10): Identifying legend-area layout...')
    #kernel = np.ones((5, 5), np.uint8)
    #floodfill_block_erosion = cv2.erode(floodfill_block, kernel, iterations = 1)
    
    kernel = np.ones((1, 15), np.uint8)
    floodfill_block_erosion = cv2.dilate(floodfill_block_erosion, kernel, iterations = 1)

    # Adaptively remove noisy columns
    kernel = np.ones((50, 1), np.uint8)
    column_summary = cv2.dilate(floodfill_block_erosion, kernel, iterations = 40)
    if np.mean(column_summary) > 0:
        column_summary_cand = np.copy(column_summary)
        column_summary_cand_prev = np.copy(column_summary_cand)
        for relaxation in range(5, 60, 5):
            kernel = np.ones((1, relaxation), np.uint8)
            column_summary_cand = cv2.erode(column_summary_cand, kernel, iterations = 1)
            column_summary_cand = cv2.dilate(column_summary_cand, kernel, iterations = 1)
            #print(np.mean(cv2.bitwise_and(roi_poly, column_summary_cand)) / np.mean(cv2.bitwise_and(roi_poly, column_summary_cand_prev)))
            if np.mean(cv2.bitwise_and(roi_poly, column_summary_cand)) / np.mean(cv2.bitwise_and(roi_poly, column_summary_cand_prev)) < 0.66:
                column_summary_cand = np.copy(column_summary_cand_prev)
                break
            column_summary_cand_prev = np.copy(column_summary_cand)
        column_summary = np.copy(column_summary_cand)
    #kernel = np.ones((1, 5), np.uint8) #
    #column_summary = cv2.erode(column_summary, kernel, iterations = 1) #
    #column_summary = cv2.dilate(column_summary, kernel, iterations = 1) #
    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_column.tif')), column_summary)






    if adaptive_processing == True and placeholder_generated == False:
        poly_candidate = cv2.bitwise_and(floodfill_block, column_summary)
        kernel = np.ones((3, 3), np.uint8)
        poly_candidate = cv2.erode(poly_candidate, kernel, iterations = 1)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly.tif')), poly_candidate)


        ######
        poly_candidate_sweeping = np.copy(poly_candidate)
        for sweeping in range(0, 300):
            kernel = np.ones((1, 2), np.uint8)
            poly_candidate_sweeping = cv2.dilate(poly_candidate_sweeping, kernel, iterations = 1)
            poly_candidate_sweeping = cv2.bitwise_and(poly_candidate_sweeping, column_summary)
        #kernel = np.ones((5, 5), np.uint8) #
        #poly_candidate_sweeping = cv2.erode(poly_candidate_sweeping, kernel, iterations = 1) #
        poly_candidate_sweeping = cv2.bitwise_and(poly_candidate_sweeping, cv2.bitwise_or(roi_poly, roi_ptln))
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly1.tif')), poly_candidate_sweeping)



        ### Filter out legend items based on areas...
        poly_contours, poly_hierarchy = cv2.findContours(poly_candidate_sweeping, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_contour_size = 50000 # 1% of image
        min_contour_size = 200
        min_bounding_size = 10

        poly_valid_contours = []
        i = 0
        # list for storing names of shapes
        for c in poly_contours:
            # here we are ignoring first counter because
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue

            # Discard shapes that are too large or small to be a valid legend
            area = cv2.contourArea(c)
            if area <= min_contour_size or area >= max_contour_size:
                continue

            # Discard shapes that are quads but not rectangular
            x,y,w,h = cv2.boundingRect(c)
            if h <= min_bounding_size or w <= min_bounding_size:
                continue

            poly_valid_contours.append(c)

        poly_img_0 = np.ones((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), dtype='uint8')

        # list for storing names of shapes
        for i in range(0,len(poly_valid_contours)):
            # using drawContours() function
            cv2.drawContours(poly_img_0, [poly_valid_contours[i]], 0, (0, 0, 255), 1)
        poly_img_0 = cv2.cvtColor(poly_img_0, cv2.COLOR_RGB2GRAY)

        # flood fill background to find inner holes
        floodfill_candidate = np.ones((poly_img_0.shape[0], poly_img_0.shape[1]), dtype='uint8') * 255
        holes = np.copy(poly_img_0)
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
        floodfill_block = cv2.bitwise_and(255-poly_img_0, valid_holes)
        
        poly_candidate_sweeping = np.copy(floodfill_block)
        poly_candidate_sweeping[poly_candidate_sweeping > 0] = 255 # 254
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly1_2.tif')), poly_candidate_sweeping)


    


    if True:
        print('Updating approach...')

        # Define thresholds for saturation and value to exclude black and white
        # Note: Adjust these thresholds according to your specific needs
        saturation_threshold_low = 30   # Lower bound for saturation
        value_threshold_low = 90        # Lower bound for value
        value_threshold_high = 225      # Upper bound for value
        # Black colors can generally be characterized by very low value, while white colors can be characterized by very high value and low saturation.

        # Create masks based on the thresholds, extracting colored areas (excluding black and white)
        threshold = cv2.inRange(img_hsv, (0, saturation_threshold_low, value_threshold_low), (180, 255, value_threshold_high))

        # find their boundaries using Canny
        threshold_dilate = cv2.Canny(threshold, 10, 20)
        kernel = np.ones((5, 5), np.uint8)
        threshold_dilate = cv2.dilate(threshold_dilate, kernel, iterations = 1)
        threshold_dilate = cv2.erode(threshold_dilate, kernel, iterations = 1)

        # flood fill background to find inner holes
        floodfill_candidate = np.ones((threshold_dilate.shape[0], threshold_dilate.shape[1]), dtype='uint8') * 255
        holes = np.copy(threshold_dilate)
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
        #floodfill_block = cv2.bitwise_and(255-threshold, valid_holes)
        floodfill_block_original = cv2.erode(valid_holes, kernel, iterations = 1)

        # combine floodfill_output (fill-up based on boundaries) with threshold_dilate_output (fill-up based on thresholding)
        kernel = np.ones((10, 15), np.uint8)
        threshold_dilate_eroded = cv2.erode(threshold_dilate, kernel, iterations = 1)
        floodfill_block = cv2.bitwise_or(floodfill_block_original, threshold_dilate_eroded)
        
        # identify areas that can be used for the triangular/ irregular polygon legend items, by excluding areas of the existing-identified polygon legend items
        kernel = np.ones((5, 10), np.uint8)
        inv_taboo = 255-cv2.dilate(poly_candidate_sweeping, kernel, iterations = 1)
        inv_taboo = cv2.bitwise_and(roi_poly, inv_taboo)

        triangular_candidate = cv2.bitwise_and(inv_taboo, floodfill_block)
        kernel = np.ones((5, 3), np.uint8)
        poly_tri_candidate_sweeping = cv2.erode(triangular_candidate, kernel, iterations = 1)

        # turn the triangular/ irregular polygon legend items into rectangular shape
        kernel = np.ones((1, 3), np.uint8)
        for sweeping in range(0, 200):
            poly_tri_candidate_sweeping = cv2.dilate(poly_tri_candidate_sweeping, kernel, iterations = 1)
            poly_tri_candidate_sweeping = cv2.bitwise_and(poly_tri_candidate_sweeping, column_summary)
        kernel = np.ones((1, 5), np.uint8)
        poly_tri_candidate_sweeping = cv2.erode(poly_tri_candidate_sweeping, kernel, iterations = 1)
        
        # extend the height of polygon legend items as much as possible
        kernel = np.ones((5, 10), np.uint8)
        inv_hieght_taboo = 255-cv2.dilate(poly_tri_candidate_sweeping, kernel, iterations = 1)
        inv_hieght_taboo = cv2.bitwise_and(inv_taboo, inv_hieght_taboo)
        kernel = np.ones((15, 30), np.uint8)
        inv_hieght_taboo = cv2.dilate(inv_hieght_taboo, kernel, iterations = 1)

        kernel = np.ones((4, 1), np.uint8)
        poly_tri_candidate_sweeping_cand = cv2.dilate(poly_tri_candidate_sweeping, kernel, iterations = 1)
        poly_tri_candidate_sweeping_cand = cv2.bitwise_and(poly_tri_candidate_sweeping_cand, inv_hieght_taboo)
        poly_tri_candidate_sweeping = cv2.bitwise_or(poly_tri_candidate_sweeping, poly_tri_candidate_sweeping_cand)

        
        #adaptive_threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_0.tif')), threshold)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_1.tif')), threshold_dilate)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_2.tif')), holes)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_3.tif')), floodfill_block_original)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_4.tif')), floodfill_block)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_5.tif')), triangular_candidate)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_6.tif')), poly_tri_candidate_sweeping)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_candidate_7.tif')), poly_tri_candidate_sweeping_cand)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_triangular_taboo_0.tif')), inv_hieght_taboo)



        ### Filter out legend items based on areas...
        poly_contours, poly_hierarchy = cv2.findContours(poly_tri_candidate_sweeping, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_contour_size = 50000 # 1% of image
        min_contour_size = 200
        min_bounding_size = 10

        poly_valid_contours = []
        i = 0
        # list for storing names of shapes
        for c in poly_contours:
            # here we are ignoring first counter because
            # findcontour function detects whole image as shape
            #if i == 0:
            #    i = 1
            #    continue

            # Discard shapes that are too large or small to be a valid legend
            area = cv2.contourArea(c)
            if area <= min_contour_size or area >= max_contour_size:
                print('get 0...')
                continue

            # Discard shapes that are quads but not rectangular
            x,y,w,h = cv2.boundingRect(c)
            if h <= min_bounding_size or w <= min_bounding_size:
                print('get 1...')
                continue

            poly_valid_contours.append(c)

        poly_img_0 = np.ones((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), dtype='uint8')

        # list for storing names of shapes
        for i in range(0,len(poly_valid_contours)):
            # using drawContours() function
            cv2.drawContours(poly_img_0, [poly_valid_contours[i]], 0, (0, 0, 255), 1)
        poly_img_0 = cv2.cvtColor(poly_img_0, cv2.COLOR_RGB2GRAY)

        # flood fill background to find inner holes
        floodfill_candidate = np.ones((poly_img_0.shape[0], poly_img_0.shape[1]), dtype='uint8') * 255
        holes = np.copy(poly_img_0)
        cv2.floodFill(holes, None, (0, 0), 255)

        # invert holes mask, bitwise or with img fill in holes
        holes = cv2.bitwise_not(holes)
        valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
        floodfill_block = cv2.bitwise_and(255-poly_img_0, valid_holes)
        
        poly_candidate_sweeping_v2 = np.copy(floodfill_block)
        poly_candidate_sweeping_v2[poly_candidate_sweeping_v2 > 0] = 255 # 254


        # make sure the new legend item does not overlap with existing legend items
        kernel = np.ones((3, 3), np.uint8)
        poly_candidate_sweeping_v2 = cv2.bitwise_and(poly_candidate_sweeping_v2, 255-cv2.dilate(poly_candidate_sweeping, kernel, iterations = 1))
        poly_candidate_sweeping_v2[poly_candidate_sweeping_v2 > 0] = 255

        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_candidate_poly2.tif')), poly_candidate_sweeping_v2)

        poly_candidate_sweeping = cv2.bitwise_or(poly_candidate_sweeping, poly_candidate_sweeping_v2)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate3'), target_map_name.replace('.tif', '_candidate_poly3.tif')), poly_candidate_sweeping)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly2.tif')), poly_candidate_sweeping)




    if  only_poly == True:
        print('============= You opt out extracting point and line legend items at this stage. Thanks for saving our times...')
        print('============= You can not bypass extraction for point and line legend items at this stage...')

        ptln_candidate_sweeping = np.zeros((column_summary.shape[0], column_summary.shape[1]), dtype='uint8')
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.tif')), ptln_candidate_sweeping)
    else:
        ptln_candidate_sweeping = np.zeros((column_summary.shape[0], column_summary.shape[1]), dtype='uint8')
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.tif')), ptln_candidate_sweeping)
    

        layout_estimated_column = False
        if np.mean(column_summary) > 0:
            anti_column_summary = 255 - column_summary
            anti_column_summary = cv2.bitwise_and(roi_poly, anti_column_summary)
            anti_column_summary = cv2.bitwise_and(anti_column_summary, adaptive_threshold)

            kernel = np.ones((1, 10), np.uint8)
            anti_column_summary = cv2.erode(anti_column_summary, kernel, iterations = 1)
            cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_anti_column.tif')), anti_column_summary)

            kernel = np.ones((50, 1), np.uint8)
            anti_column_summary = cv2.dilate(anti_column_summary, kernel, iterations = 40)
            cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_anti_column2.tif')), anti_column_summary)

            kernel = np.ones((1, 5), np.uint8)
            column_summary_buffer = cv2.dilate(column_summary, kernel, iterations = 1)
            overlapping_column_summary = cv2.bitwise_and(column_summary_buffer, anti_column_summary)
            cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_overlapping_column.tif')), overlapping_column_summary)

            anti_column_summary = cv2.bitwise_and(anti_column_summary, 255-column_summary_buffer)
            cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_anti_column3.tif')), anti_column_summary)

            layout_estimated_column = True
            if np.mean(cv2.bitwise_and(column_summary_buffer, roi_poly)) > 0:
                overlapping_column_area = np.mean(cv2.bitwise_and(overlapping_column_summary, roi_poly)) / np.mean(cv2.bitwise_and(column_summary_buffer, roi_poly))
                #print(overlapping_column_area)
                if overlapping_column_area > 0.9:
                    layout_estimated_column = False
        else:
            anti_column_summary = np.zeros((column_summary.shape[0], column_summary.shape[1]), dtype='uint8')

        if layout_estimated_column == True:
            print('------------- Assumed columned-based layout...')
        else:
            print('------------- Assumed layout without enough information...')
        


        if layout_estimated_column == True and candidate_width.shape[0] > 0:
            # Trying to find the column if exists...
            '''
            kernel = np.ones((50, 1), np.uint8)
            column_summary_buffer_v2 = cv2.dilate(column_summary_buffer, kernel, iterations = 40)
            '''

            original_column_area = np.mean(column_summary)
            prev_column_area = original_column_area

            extanded_column_area = []
            extanded_column_area.append(original_column_area)
            column_summary_extand = np.copy(column_summary)
            column_summary_extand_settled = np.copy(column_summary)
            extanded_i = -1
            jumping_i = 0
            for i in range(2, 20):
                if jumping_i > 0:
                    jumping_i -= 1
                    continue
                
                print('------------- I am testing a step that jumps '+str(int(median_width) * i)+' pixels...')
                if (median_width * i) > img_gray.shape[1]:
                    break
                for j in range(int(median_width) * i, int(median_width) * (i+1)):
                    column_summary_extand[:, j:] = np.maximum(column_summary[:, j:], column_summary[:, 0:-j]) # growing right
                this_column_area = np.mean(column_summary_extand)
                if this_column_area < prev_column_area * 0.9:
                    print('------------- columned-based ROI extanded...')
                    column_summary_extand_settled = np.copy(column_summary_extand)
                    cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_column_expanded.tif')), column_summary_extand_settled)
                    if extanded_i == -1:
                        extanded_i = i
                    if extanded_i > 0:
                        extanded_column_area.append(this_column_area)

                        rows, cols = np.where(column_summary_extand > 0)
                        if len(cols) > 0:
                            max_col_index = np.max(cols)
                        else:
                            max_col_index = 0
                        #print(max_col_index)

                        if (max_col_index + median_width * (extanded_i)) > img_gray.shape[1]:
                            break
                        else:
                            jumping_i = extanded_i-2
                            prev_column_area = this_column_area
                            continue
                extanded_column_area.append(this_column_area)
                prev_column_area = this_column_area
            print('------------- ', extanded_column_area)
            column_summary = np.copy(column_summary_extand_settled)


        '''
        print('Step ( 6/10): Reading results from mapkurator...')
        #fast_processing = True
        #if fast_processing == True and os.path.isfile(os.path.join(path_to_intermediate, 'intermediate3', target_map_name.replace('.tif', '_mapkurator_mask_buffer_v1.tif'))) == True:
            #print('fast processing for text spotting...')
            #pass
        #else:
        if True:
            read_results_from_mapkurator(target_map_name, path_to_intermediate, path_to_mapkurator_output)
        


        print('Step ( 7/10): Identifying legend items for point and line features...')
        text_spotting_mask = cv2.imread(os.path.join(path_to_intermediate, 'intermediate3', target_map_name.replace('.tif', '_mapkurator_mask_buffer_v1.tif')))
        text_spotting_mask = cv2.cvtColor(text_spotting_mask, cv2.COLOR_BGR2GRAY)

        kernel = np.ones((1, 100), np.uint8)
        text_spotting_mask = cv2.erode(text_spotting_mask, kernel, iterations = 1)
        text_spotting_mask = cv2.dilate(text_spotting_mask, kernel, iterations = 1)
        cv2.imwrite(os.path.join(path_to_intermediate, 'intermediate3', target_map_name.replace('.tif', '_mapkurator_mask_buffer_v2.tif')), text_spotting_mask)
        '''
        

        if np.mean(column_summary) == 0:
            print('------------- There is no column summary identified...')
            print('------------- An additional input from text-spotting is highly recommended...')
            print('------------- ......')

            print('Step (?6/10): Reading results from mapkurator...')
            read_results_from_mapkurator(target_map_name, path_to_intermediate, path_to_mapkurator_output)

            text_spotting_mask = cv2.imread(os.path.join(path_to_intermediate, 'intermediate3', target_map_name.replace('.tif', '_mapkurator_mask_buffer_v1.tif')))
            text_spotting_mask = cv2.cvtColor(text_spotting_mask, cv2.COLOR_BGR2GRAY)

            kernel = np.ones((1, 100), np.uint8)
            text_spotting_mask = cv2.erode(text_spotting_mask, kernel, iterations = 1)
            text_spotting_mask = cv2.dilate(text_spotting_mask, kernel, iterations = 1)
            cv2.imwrite(os.path.join(path_to_intermediate, 'intermediate3', target_map_name.replace('.tif', '_mapkurator_mask_buffer_v2.tif')), text_spotting_mask)

            column_summary = 255-text_spotting_mask
        



        print('Step ( 7/10): Identifying legend items for point and line features...')
        kernel = np.ones((2, 2), np.uint8)
        adaptive_buffer = cv2.erode(adaptive_threshold, kernel, iterations = 1)
        kernel = np.ones((1, 10), np.uint8)
        column_summary = cv2.erode(column_summary, kernel, iterations = 1)
        adaptive_buffer = cv2.bitwise_and(column_summary, adaptive_buffer)
            
        kernel = np.ones((5, 5), np.uint8)
        adaptive_buffer = cv2.dilate(adaptive_buffer, kernel, iterations = 1)
        ptln_candidate = cv2.bitwise_and(roi_ptln, adaptive_buffer)
        
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_preliminary_ptln.tif')), ptln_candidate)
        ptln_candidate = cv2.erode(ptln_candidate, kernel, iterations = 1)



        # clean the nearby areas in advance for polygons that we get from extracting poly legend items
        kernel = np.ones((10, 15), np.uint8)
        ptln_candidate_from_poly = cv2.bitwise_and(poly_candidate_sweeping, roi_ptln)
        ptln_candidate_from_poly_buffer = cv2.dilate(ptln_candidate_from_poly, kernel, iterations = 1)
        ptln_candidate_from_poly_buffer = cv2.bitwise_and(ptln_candidate_from_poly_buffer, roi_ptln)
        ptln_candidate = cv2.bitwise_and(ptln_candidate, 255-ptln_candidate_from_poly_buffer)

        kernel = np.ones((10, 10), np.uint8)
        ptln_candidate = cv2.dilate(ptln_candidate, kernel, iterations = 1)
        ptln_candidate = cv2.erode(ptln_candidate, kernel, iterations = 1)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_preliminary_ptln1.tif')), ptln_candidate)


        kernel = np.ones((25, 50), np.uint8)
        ptln_candidate_sweeping = cv2.dilate(ptln_candidate, kernel, iterations = 1)
        ptln_candidate_sweeping = cv2.bitwise_and(column_summary, ptln_candidate_sweeping)
        
        # include polygons that we get from extracting poly legend items
        ptln_candidate_from_poly = cv2.bitwise_and(poly_candidate_sweeping, roi_ptln)
        ptln_candidate_sweeping = cv2.bitwise_or(ptln_candidate_sweeping, ptln_candidate_from_poly)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_preliminary_ptln2.tif')), ptln_candidate_sweeping)



        # extending the width of point and line legend items
        kernel = np.ones((25, 1), np.uint8)
        ptln_column_summary = cv2.dilate(ptln_candidate_sweeping, kernel, iterations = 40)
        

        kernel = np.ones((1, 3), np.uint8)
        for sweeping in range(0, 200):
            ptln_candidate_sweeping = cv2.dilate(ptln_candidate_sweeping, kernel, iterations = 1)
            #ptln_candidate_sweeping = cv2.bitwise_and(ptln_candidate_sweeping, column_summary_buffer_v2)
            ptln_candidate_sweeping = cv2.bitwise_and(ptln_candidate_sweeping, ptln_column_summary)
        
        kernel = np.ones((3, 1), np.uint8)
        ptln_candidate_sweeping = cv2.erode(ptln_candidate_sweeping, kernel, iterations = 1)
        ptln_candidate_sweeping = cv2.dilate(ptln_candidate_sweeping, kernel, iterations = 1)
        ptln_candidate_sweeping[ptln_candidate_sweeping > 0] = 255

        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln1.tif')), ptln_candidate_sweeping)



        ### Filter out legend items based on areas...
        ptln_contours, ptln_hierarchy = cv2.findContours(ptln_candidate_sweeping, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_contour_size = 50000 # 1% of image
        min_contour_size = 8000

        ptln_candidate_areas = []
        ptln_candidate_contours = []
        i = 0
        # list for storing names of shapes
        for c in ptln_contours:
            # here we are ignoring first counter because
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue

            # Discard shapes that are too large or small to be a valid legend
            area = cv2.contourArea(c)
            if area <= min_contour_size or area >= max_contour_size:
                continue

            # Discard shapes that are quads but not rectangular
            x,y,w,h = cv2.boundingRect(c)

            ptln_candidate_areas.append(area)
            ptln_candidate_contours.append(c)

        # Calculate valid area threshold
        ptln_candidate_df = pd.DataFrame({'area' : ptln_candidate_areas, 'contour' :ptln_candidate_contours})
        ptln_candidate_areas.sort()

        if len(ptln_candidate_areas) > 0:
            ptln_median_area = np.median(ptln_candidate_areas)
            ptln_mmin_valid_area = ptln_median_area - ptln_median_area*0.66
            if candidate_width.shape[0] > 0:
                ptln_mmax_valid_area = max(ptln_median_area*10.0, median_rect_area + median_rect_area*0.66)
            else:
                ptln_mmax_valid_area = ptln_median_area*20.0
            
            
            ptln_valid_contours = []
            for c in ptln_candidate_contours:
                # Discard shapes outside the valid contour areas
                area = cv2.contourArea(c)
                if area <= ptln_mmin_valid_area or area >= ptln_mmax_valid_area:
                    continue

                ptln_valid_contours.append(c)
            

            ptln_img = np.ones((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), dtype='uint8')

            # list for storing names of shapes
            for i in range(0,len(ptln_valid_contours)):
                # using drawContours() function
                cv2.drawContours(ptln_img, [ptln_valid_contours[i]], 0, (0, 0, 255), 1)
            ptln_img = cv2.cvtColor(ptln_img, cv2.COLOR_RGB2GRAY)

            # flood fill background to find inner holes
            floodfill_candidate = np.ones((ptln_img.shape[0], ptln_img.shape[1]), dtype='uint8') * 255
            holes = np.copy(ptln_img)
            cv2.floodFill(holes, None, (0, 0), 255)

            # invert holes mask, bitwise or with img fill in holes
            holes = cv2.bitwise_not(holes)
            valid_holes = cv2.bitwise_and(holes, floodfill_candidate)
            floodfill_block = cv2.bitwise_and(255-ptln_img, valid_holes)
            
            ptln_candidate_sweeping = np.copy(floodfill_block)
        
        ptln_candidate_sweeping[ptln_candidate_sweeping > 0] = 255 # 254
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.tif')), ptln_candidate_sweeping)

        '''
        ### rectangularization
        # tif > geojson
        base_image = cv2.imread(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.tif')))
        input_raster = os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.png'))
        output_vector = os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.geojson'))
        cv2.imwrite(input_raster, base_image)

        in_path = input_raster
        out_path = output_vector

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

        polygon_extraction = gpd.read_file(output_vector, driver='GeoJSON')
        mirrored_polygon = gpd.GeoDataFrame(columns=['id', 'geometry'], crs=polygon_extraction.crs)
        for index, poi in polygon_extraction.iterrows():
            if index == polygon_extraction.shape[0]-1:
                break

            # convert to bounding box
            bbox_tuple = poi['geometry'].bounds
            geom = shapely.geometry.box(*bbox_tuple)
            #print(this_mirrored_polygon)
            #print(bbox_tuple, ' >>> ', geom)
            this_mirrored_polygon = geom

            updated_record = gpd.GeoDataFrame([{'id': index, 'geometry':this_mirrored_polygon}])
            mirrored_polygon = gpd.GeoDataFrame(pd.concat( [mirrored_polygon, updated_record], ignore_index=True), crs=polygon_extraction.crs)

        mirrored_polygon = mirrored_polygon.set_crs('epsg:3857', allow_override=True)
        mirrored_polygon.to_file(output_vector.replace('_candidate_ptln2.geojson', '_candidate_ptln3.geojson'), driver='GeoJSON')


        # geojson > tif
        source_name = output_vector.replace('_candidate_ptln2.geojson', '_candidate_ptln3.geojson')
        basemap_name = os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.tif'))

        base_image = cv2.imread(basemap_name)
        base_image = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

        gdf = gpd.read_file(source_name, driver='GeoJSON')
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
            if poi['area'] > (base_image.shape[0] * base_image.shape[1]) * 0.8:
                continue
            this_text_mask = rasterio.features.rasterize([gdf.loc[index]['geometry']], out_shape=(base_image.shape[0], base_image.shape[1]))
            text_mask = cv2.bitwise_or(text_mask, this_text_mask)
            
        text_mask[text_mask > 0] = 255

        kernel = np.ones((3, 3), np.uint8)
        ptln_candidate_rectangularized = cv2.dilate(text_mask, kernel, iterations = 1)
        ptln_candidate_rectangularized = cv2.bitwise_and(ptln_block, ptln_candidate_rectangularized)
        cv2.imwrite(os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln3.tif')), ptln_candidate_rectangularized)
        '''



    if only_poly == True:
        integrated_raster_legend_item = np.copy(poly_candidate_sweeping)
    else:
        integrated_raster_legend_item = cv2.bitwise_or(poly_candidate_sweeping, ptln_candidate_sweeping)
    cv2.imwrite(os.path.join(path_to_intermediate, target_map_name.replace('.tif', '_integrated_raster.tif')), integrated_raster_legend_item)

    integrated_raster_legend_item_bounding_box = np.copy(integrated_raster_legend_item)
    integrated_raster_legend_item_bounding_box[integrated_raster_legend_item_bounding_box > 0] = 255

    kernel = np.ones((15, 15), np.uint8)
    integrated_raster_legend_item_bounding_box_erode = cv2.erode(integrated_raster_legend_item_bounding_box, kernel, iterations = 1)
    integrated_raster_legend_item_bounding_box = cv2.bitwise_and(integrated_raster_legend_item_bounding_box, 255-integrated_raster_legend_item_bounding_box_erode)

    red_place_holder = np.zeros((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), dtype='uint8')
    color = tuple(reversed((0, 0, 255)))
    red_place_holder[:] = color

    black_place_holder = np.zeros((img_rgb.shape[0], img_rgb.shape[1], img_rgb.shape[2]), dtype='uint8')
    #integrated_raster_visualization = cv2.copyTo(src=img_rgb, dst=black_place_holder, mask=integrated_raster_legend_item)

    #integrated_raster_visualization = cv2.copyTo(src=red_place_holder, dst=integrated_raster_visualization, mask=integrated_raster_legend_item_bounding_box)
    integrated_raster_visualization = cv2.copyTo(src=red_place_holder, dst=img_rgb, mask=integrated_raster_legend_item_bounding_box)
    integrated_raster_visualization = cv2.cvtColor(integrated_raster_visualization, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(path_to_intermediate, target_map_name.replace('.tif', '_integrated_raster_vis.tif')), integrated_raster_visualization)













def start_linking(target_map_name, input_image, output_dir, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, preprocessing_for_cropping, postprocessing_for_crs, competition_custom, placeholder_handler, only_poly, version):
    flag_identify, placeholder_generated = map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, placeholder_handler, competition_custom)
    
    if flag_identify == False:
        with open('missing.csv','a') as fd:
            fd.write(str(target_map_name)+'\n')
            fd.close()
            return False
    
    if version == '1':
        print('Version 1.2 is deployed...')
        exit(1)
    elif version == '1.2':
        if flag_identify == True:
            map_key_extraction(target_map_name, path_to_intermediate, path_to_mapkurator_output, placeholder_generated, only_poly)
        else:
            with open('missing.csv','a') as fd:
                fd.write(str(target_map_name)+'\n')
                fd.close()
    return True


def main():
    #flag_skipping = False
    
    this_map_count = 0
    total_map_count = len(os.listdir('Data/testing/'))/2
    for target_map_cand in os.listdir('Data/testing/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/testing/'+str(target_map)+'.tif'
            path_to_intermediate = 'Example_Output/LINK_Intermediate/testing/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-evaluation/'+str(target_map)+'.geojson'

            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True, False, '1.2')
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
            this_map_count += 1


    this_map_count = 0
    total_map_count = len(os.listdir('Data/validation/'))/2
    for target_map_cand in os.listdir('Data/validation/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/validation/'+str(target_map)+'.tif'
            path_to_intermediate = 'Example_Output/LINK_Intermediate/validation/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-validation/'+str(target_map)+'.geojson'

            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True, False, '1.2')
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
            this_map_count += 1


    this_map_count = 0
    total_map_count = len(os.listdir('Data/training/'))/2
    for target_map_cand in os.listdir('Data/training/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/training/'+str(target_map)+'.tif'
            path_to_intermediate = 'Example_Output/LINK_Intermediate/training/'+str(target_map)+'/'
            input_area_segmentation = None
            input_legend_segmentation = 'Uncharted/ch2_training_labels_coco.json'
            path_to_mapkurator_output = 'MapKurator/ta1-feature-training/'+str(target_map)+'.geojson'

            os.makedirs(os.path.dirname(path_to_intermediate), exist_ok=True)

            start_linking(target_map_name, input_image, None, path_to_intermediate, input_area_segmentation, input_legend_segmentation, path_to_mapkurator_output, None, None, True, False, '1.2')
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
            this_map_count += 1

    

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()





