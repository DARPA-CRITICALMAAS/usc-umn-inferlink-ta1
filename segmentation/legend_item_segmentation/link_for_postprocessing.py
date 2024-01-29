
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
import csv





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
            legend_area_placeholder0 = np.zeros((this_gj['height'], this_gj['width']), dtype='uint8')
            legend_area_placeholder1 = np.zeros((this_gj['height'], this_gj['width']), dtype='uint8')
            break
    
    if target_id == -1:
        return False

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
            
            if segmentation_added >= 2:
                break
    return True


def processing_uncharted_json_single(input_image, input_legend_segmentation, target_map_name, output_segmentation):
    with open(input_legend_segmentation) as f:
        gj = json.load(f)
    
    img0 = cv2.imread(input_image)
    gray0 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    legend_area_placeholder = np.zeros((gray0.shape[0],gray0.shape[1]), dtype='uint8')
    for this_gj in gj['segments']:
        if 'legend_polygons' in this_gj['class_label']:
            cv2.fillConvexPoly(legend_area_placeholder, np.array(this_gj['poly_bounds']), 1)
            legend_area_placeholder[legend_area_placeholder > 0] = 255
            cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_poly.tif'), legend_area_placeholder)
            break

    legend_area_placeholder = np.zeros((gray0.shape[0],gray0.shape[1]), dtype='uint8')
    for this_gj in gj['segments']:
        if 'legend_points_lines' in this_gj['class_label']:
            cv2.fillConvexPoly(legend_area_placeholder, np.array(this_gj['poly_bounds']), 1)
            legend_area_placeholder[legend_area_placeholder > 0] = 255
            cv2.imwrite(output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif'), legend_area_placeholder)
            break
    return True
                    

def map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, competition_custom):
    #print('Working on input image:', input_image)

    output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')
    flag_identify = False

    if len(input_legend_segmentation) > 0:
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
    
    return True



def reading_model_based_output(target_map_name, path_to_intermediate):
    dir_pred_testing1 = 'LOAM_Intermediate/predict/cma/'
    output_segmentation = os.path.join(path_to_intermediate, 'exc_crop_binary.tif')

    model_based_output_for_poly = dir_pred_testing1 + target_map_name.replace('.tif', '_predict.png')
    model_based_output_for_ptln = dir_pred_testing1 + target_map_name.replace('.tif', '_predict2.png')

    legend_area_candidate_poly = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_poly.tif')
    legend_area_candidate_ptln = output_segmentation.replace('exc_crop_binary.tif', 'area_crop_ptln.tif')

    raster_output_for_poly = cv2.imread(model_based_output_for_poly)
    raster_output_for_poly = cv2.cvtColor(raster_output_for_poly, cv2.COLOR_BGR2GRAY)
    raster_output_for_ptln = cv2.imread(model_based_output_for_ptln)
    raster_output_for_ptln = cv2.cvtColor(raster_output_for_ptln, cv2.COLOR_BGR2GRAY)
    
    placeholder = np.zeros((raster_output_for_poly.shape[0], raster_output_for_poly.shape[1]), dtype='uint8')
    if os.path.isfile(legend_area_candidate_poly) == False:
        cv2.imwrite(legend_area_candidate_poly, placeholder)
    if os.path.isfile(legend_area_candidate_ptln) == False:
        cv2.imwrite(legend_area_candidate_ptln, placeholder)

    raster_area_for_poly = cv2.imread(legend_area_candidate_poly)
    raster_area_for_poly = cv2.cvtColor(raster_area_for_poly, cv2.COLOR_BGR2GRAY)
    raster_area_for_ptln = cv2.imread(legend_area_candidate_ptln)
    raster_area_for_ptln = cv2.cvtColor(raster_area_for_ptln, cv2.COLOR_BGR2GRAY)

    raster_output_for_poly = cv2.bitwise_and(raster_output_for_poly, raster_area_for_poly)
    raster_output_for_ptln = cv2.bitwise_and(raster_output_for_ptln, raster_area_for_ptln)

    kernel = np.ones((5, 5), np.uint8)
    raster_output_for_poly = cv2.dilate(raster_output_for_poly, kernel, iterations = 1)

    kernel = np.ones((10, 30), np.uint8)
    raster_output_for_ptln = cv2.dilate(raster_output_for_ptln, kernel, iterations = 1)


    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate7'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate7')))
    cv2.imwrite(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_poly.tif')), raster_output_for_poly)
    cv2.imwrite(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_ptln.tif')), raster_output_for_ptln)



def compiling_geojson(target_map_name, input_image, output_dir, path_to_intermediate):
    print('Step (10/10): Preparing output json - Generating GEOJSON file (GPKG schema)...')
    legend_item_counter = 0

    in_path = os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_poly.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_poly.geojson'))

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


    layer1 = gpd.read_file(out_path, driver='GeoJSON')


    img0 = cv2.imread(input_image)
    rgb0 = cv2.cvtColor(img0,cv2.COLOR_BGR2RGB)
    hsv0 = cv2.cvtColor(img0,cv2.COLOR_BGR2HSV)

    linked_poly_description = gpd.GeoDataFrame(columns=['name', 'abbreviation', 'id', 'map_unit', 'color', 'pattern', 'description', 'category', 'geometry'], crs=layer1.crs)
    
    for index, row in layer1.iterrows():
        # Get color and pattern of polygonal legend item...
        #xmin, ymin, xmax, ymax = shapely.wkt.loads(row['geometry']).bounds
        xmin, ymin, xmax, ymax = gpd.GeoSeries(row['geometry']).values[0].bounds
        x_delta = xmax - xmin
        y_delta = ymax - ymin

        if (x_delta*y_delta) > (rgb0.shape[0]*rgb0.shape[1])*0.8:
            continue

        this_map_key = rgb0[int(ymin+y_delta*0.2): max(int(ymax-y_delta*0.2), int(ymin+y_delta*0.2)+2), int(xmin+x_delta*0.2): max(int(xmax-x_delta*0.2), int(xmin+x_delta*0.2)+2), :]
        this_map_key_hsv = hsv0[int(ymin+y_delta*0.2): max(int(ymax-y_delta*0.2), int(ymin+y_delta*0.2)+2), int(xmin+x_delta*0.2): max(int(xmax-x_delta*0.2), int(xmin+x_delta*0.2)+2), :]

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

        #print(this_map_key.shape)
        #print(rgb_trimmed.shape)
        #print(np.sum(np.isnan(rgb_trimmed)))
        #print((rgb_trimmed.shape[0]*rgb_trimmed.shape[1]*rgb_trimmed.shape[2]))
        if np.sum(np.isnan(rgb_trimmed)) >= (rgb_trimmed.shape[0]*rgb_trimmed.shape[1]*rgb_trimmed.shape[2]):
            median_color = np.array([int(np.nanquantile(rgb_trimmed_source[0],.5)),int(np.nanquantile(rgb_trimmed_source[1],.5)),int(np.nanquantile(rgb_trimmed_source[2],.5))])
        else:
            median_color = np.array([int(np.nanquantile(rgb_trimmed[0],.5)),int(np.nanquantile(rgb_trimmed[1],.5)),int(np.nanquantile(rgb_trimmed[2],.5))])
        this_hex = rgb2hex(median_color/255)


        complexity = ''
        hsv_trimmed = np.zeros((this_map_key_hsv.shape[2], this_map_key_hsv.shape[0], this_map_key_hsv.shape[1]), dtype='uint8')
        hsv_trimmed = hsv_trimmed.astype(float)
        for dimension in range(0, 3):
            hsv_trimmed[dimension] = np.copy(this_map_key_hsv[:,:,dimension]).astype(float)
        lower_hsv_trimmed = np.array([int(np.nanquantile(hsv_trimmed[0],.16)),int(np.nanquantile(hsv_trimmed[1],.16)),int(np.nanquantile(hsv_trimmed[2],.16))])
        higher_hsv_trimmed = np.array([int(np.nanquantile(hsv_trimmed[0],.84)),int(np.nanquantile(hsv_trimmed[1],.84)),int(np.nanquantile(hsv_trimmed[2],.84))])
        confidence_hsv = abs(higher_hsv_trimmed[0] - lower_hsv_trimmed[0])

        if confidence_hsv < 30:
            complexity = 'solid'
        else:
            complexity = 'mixture'

        updated_record = gpd.GeoDataFrame([{'name' : '', 
                                            'abbreviation' : '', 
                                            'id' : int(legend_item_counter), 
                                            'map_unit' : None, 
                                            'color' : str(this_hex), 
                                            'pattern' : complexity, 
                                            'description' : '', 
                                            'category' : '', 
                                            'geometry' : row['geometry']
                                            }])
        legend_item_counter += 1
        linked_poly_description = gpd.GeoDataFrame(pd.concat( [linked_poly_description, updated_record], ignore_index=True), crs=layer1.crs)

    linked_poly_description1 = linked_poly_description.set_crs('epsg:3857', allow_override=True)
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_PolygonType.geojson')), driver='GeoJSON')

    for index, row in linked_poly_description1.iterrows():
        linked_poly_description1.loc[linked_poly_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['geometry']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    linked_poly_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_PolygonType_2.geojson')), driver='GeoJSON')



    ################
        
    in_path = os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_ptln.tif'))
    out_path = os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_ptln.geojson'))

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


    layer2 = gpd.read_file(out_path, driver='GeoJSON')
    linked_ptln_description = gpd.GeoDataFrame(columns=['name', 'id', 'description', 'geometry'], crs=layer2.crs)
    
    for index, row in layer2.iterrows():
        xmin, ymin, xmax, ymax = gpd.GeoSeries(row['geometry']).values[0].bounds
        x_delta = xmax - xmin
        y_delta = ymax - ymin
        if (x_delta*y_delta) > (rgb0.shape[0]*rgb0.shape[1])*0.8:
            continue

        updated_record = gpd.GeoDataFrame([{'name' : '', 
                                            'id' : int(legend_item_counter), 
                                            'description' : '', 
                                            'geometry' : row['geometry']
                                            }])
        legend_item_counter += 1
        linked_ptln_description = gpd.GeoDataFrame(pd.concat( [linked_ptln_description, updated_record], ignore_index=True), crs=layer1.crs)

    linked_ptln_description1 = linked_ptln_description.set_crs('epsg:3857', allow_override=True)
    linked_ptln_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_PointLineType.geojson')), driver='GeoJSON')

    for index, row in linked_ptln_description1.iterrows():
        linked_ptln_description1.loc[linked_ptln_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['geometry']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    linked_ptln_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_PointLineType_2.geojson')), driver='GeoJSON')





def generating_json(target_map_name, input_image, output_dir, path_to_intermediate):
    print('Step (10/10): Preparing output json - Generating JSON file (competition format)...')

    map_name = target_map_name

    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType.geojson')), driver='GeoJSON')
    new_item = []

    for index, row in linked_poly_description1.iterrows():
        this_poly_name = str(row['id']) + '_poly'
        this_legend_bounds = row['geometry'].bounds
        this_item = {}
        this_item['label'] = this_poly_name
        this_item['points'] = [[this_legend_bounds[0], -1.0*this_legend_bounds[1]], [this_legend_bounds[2], -1.0*this_legend_bounds[3]]]
        this_item['group_id'] = None
        this_item['shape_type'] = 'rectangle'
        this_item['flags']: None
        new_item.append(this_item)

    updated_record = {}
    updated_record['version'] = '1.0.1'
    updated_record['flags'] = None
    updated_record['shapes'] = new_item

    with open(os.path.join(output_dir, map_name.replace('.tif', '_PolygonType_internal.json')), 'w') as outfile: 
        json.dump(updated_record, outfile)
    outfile.close()



    linked_poly_description1 = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType.geojson')), driver='GeoJSON')
    new_item = []

    for index, row in linked_poly_description1.iterrows():
        this_poly_name = str(row['id']) + '_ptln'
        this_legend_bounds = row['geometry'].bounds
        this_item = {}
        this_item['label'] = this_poly_name
        this_item['points'] = [[this_legend_bounds[0], -1.0*this_legend_bounds[1]], [this_legend_bounds[2], -1.0*this_legend_bounds[3]]]
        this_item['group_id'] = None
        this_item['shape_type'] = 'rectangle'
        this_item['flags']: None
        new_item.append(this_item)

    updated_record = {}
    updated_record['version'] = '1.0.1'
    updated_record['flags'] = None
    updated_record['shapes'] = new_item

    with open(os.path.join(output_dir, map_name.replace('.tif', '_PointLineType_internal.json')), 'w') as outfile: 
        json.dump(updated_record, outfile)
    outfile.close()


    return True




def adjusting_crs(target_map_name, input_image, path_to_intermediate, output_dir, postprocessing_for_crs):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    map_name = target_map_name
    basemap_name = input_image

    crs_flag = False
    if postprocessing_for_crs == True:
        try:
            # convert the image to a binary raster .tif
            raster = rasterio.open(basemap_name)
            transform = raster.transform
            array     = raster.read(1)
            crs       = raster.crs 
            width     = raster.width 
            height    = raster.height 
            raster.close()

            this_epsg_code = pyproj.crs.CRS.from_proj4(crs.to_proj4()).to_epsg()
        
            trans_np = np.array(transform)
            trans_matrix = [trans_np[0], trans_np[1], trans_np[3], -trans_np[4], trans_np[2], trans_np[5]]
            #print(trans_matrix)

            original_file = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType.geojson')), driver='GeoJSON')
            for index, poi in original_file.iterrows():
                geo_series = gpd.GeoSeries(poi['geometry'])
                original_file.loc[index, 'geometry'] = geo_series.affine_transform(trans_matrix).values[0]
            original_file = original_file.set_crs('epsg:'+str(this_epsg_code), allow_override=True)
            original_file.to_file(os.path.join(output_dir, map_name.replace('.tif', '_PolygonType_crs.geojson')), driver='GeoJSON')


            original_file = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType.geojson')), driver='GeoJSON')
            for index, poi in original_file.iterrows():
                geo_series = gpd.GeoSeries(poi['geometry'])
                original_file.loc[index, 'geometry'] = geo_series.affine_transform(trans_matrix).values[0]
            original_file = original_file.set_crs('epsg:'+str(this_epsg_code), allow_override=True)
            original_file.to_file(os.path.join(output_dir, map_name.replace('.tif', '_PointLineType_crs.geojson')), driver='GeoJSON')
            
            crs_flag = True
        except:
            print('Invalid CRS...')

    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PolygonType.geojson')))
    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PointLineType.geojson')))

    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PolygonType_qgis.geojson')))
    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PointLineType_qgis.geojson')))

    print('Output at ' + str(output_dir)+' as json and geojson files...')
    print(os.path.join(output_dir, map_name.replace('.tif', '_PolygonType.geojson')), 'for legend item segmentation (polygon) (GPKG_GEOJSON format, image coordinate)')
    print(os.path.join(output_dir, map_name.replace('.tif', '_PointLineType.geojson')), 'for legend item segmentation (point, line) (GPKG_GEOJSON format, image coordinate)')
    print(os.path.join(output_dir, map_name.replace('.tif', '_[Type]_internal.json')), 'for internal usage (competition format)')
    if postprocessing_for_crs == True and crs_flag == True:
        print(os.path.join(output_dir, map_name.replace('.tif', '_[Type]_crs.geojson')), 'for output with transformed crs (map coordinate)')
    print(os.path.join(output_dir, map_name.replace('.tif', '_[Type]_qgis.geojson')), 'for output visualizable in qgis (qgis coordinate)')
    print('Legend-item Segmentation has concluded for input image:', input_image)

    return True





def start_linking_postprocessing(target_map_name, input_image, output_dir, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom):
    if '.tif' not in target_map_name:
        target_map_name = target_map_name + '.tif'
    
    print('Step ( 9/10): Postprocessing raster segmentation output to vector format...')
    flag_identify = map_area_cropping(target_map_name, input_image, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, competition_custom)
    if flag_identify == True:
        reading_model_based_output(target_map_name, path_to_intermediate)
        compiling_geojson(target_map_name, input_image, output_dir, path_to_intermediate)
        generating_json(target_map_name, input_image, output_dir, path_to_intermediate)
        adjusting_crs(target_map_name, input_image, path_to_intermediate, output_dir, postprocessing_for_crs)
    else:
        pass



def main():
    missing_list = []
    with open('missing.csv', newline='') as fdd:
        reader = csv.reader(fdd)
        for row in reader:
            missing_list.append(row[0])
    print(missing_list)


    this_map_count = 0
    total_map_count = len(os.listdir('Data/testing/'))/2
    for target_map_cand in os.listdir('Data/testing/'):
        if '.tif' in target_map_cand:
            target_map = target_map_cand.split('.tif')[0]
            if target_map_cand in missing_list:
                this_map_count += 1
                print('Disintegrity map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))
                continue

            target_map_name = str(target_map)+'.tif'
            input_image = 'Data/testing/'+str(target_map)+'.tif'
            output_dir = 'Example_Output/Vectorization_Output/'
            path_to_intermediate = 'LINK_Intermediate/testing/'+str(target_map)+'/'
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'

            os.makedirs(os.path.dirname(output_dir), exist_ok=True)

            start_linking_postprocessing(target_map_name, input_image, output_dir, path_to_intermediate, None, input_legend_segmentation, False, True, True)
            this_map_count += 1
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()