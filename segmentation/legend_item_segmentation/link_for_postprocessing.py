
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



y_stretch_factor = 1.2


def reading_raster_output(target_map_name, path_to_intermediate):
    if not os.path.exists(os.path.join(path_to_intermediate, str('intermediate7'))):
        os.makedirs(os.path.join(path_to_intermediate, str('intermediate7')))

    legend_area_candidate_poly = os.path.join(path_to_intermediate, str('intermediate4'), target_map_name.replace('.tif', '_candidate_poly2.tif'))
    legend_area_candidate_ptln = os.path.join(path_to_intermediate, str('intermediate2'), target_map_name.replace('.tif', '_candidate_ptln2.tif'))

    shutil.copyfile(legend_area_candidate_poly, os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_poly.tif')))
    shutil.copyfile(legend_area_candidate_ptln, os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_legend_ptln.tif')))



def compiling_geojson(target_map_name, input_image, output_dir, path_to_intermediate):
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

    bounding_box_height1 = []
    for index, row in layer1.iterrows():
        xmin, ymin, xmax, ymax = gpd.GeoSeries(row['geometry']).values[0].bounds
        x_delta = xmax - xmin
        y_delta = ymax - ymin
        if (x_delta*y_delta) > (rgb0.shape[0]*rgb0.shape[1])*0.8:
            continue
        bounding_box_height1.append(y_delta)

    bounding_box_height1 = np.array(bounding_box_height1)
    bounding_box_threshold1 = 20
    if bounding_box_height1.shape[0] > 0:
        bounding_box_threshold1 = min(np.quantile(bounding_box_height1, 0.5), np.mean(bounding_box_height1)) * 0.8
    #print(bounding_box_threshold2)
    
    for index, row in layer1.iterrows():
        # Get color and pattern of polygonal legend item...
        #xmin, ymin, xmax, ymax = shapely.wkt.loads(row['geometry']).bounds
        xmin, ymin, xmax, ymax = gpd.GeoSeries(row['geometry']).values[0].bounds
        x_delta = xmax - xmin
        y_delta = ymax - ymin

        # Remove large-area polygons
        if (x_delta*y_delta) > (rgb0.shape[0]*rgb0.shape[1])*0.8:
            continue
        # Remove small-area polygons (noises)
        #print(x_delta, y_delta, x_delta*y_delta)

        this_geometry = row['geometry']
        # Stretch flat bounding box
        if y_delta < bounding_box_threshold1 and y_delta > 0:
            center_x = xmin + x_delta*0.5
            center_y = ymin + y_delta*0.5

            y_stretch_factor = max(1.2, bounding_box_threshold1 * 0.5 / y_delta)

            this_geo_series = gpd.GeoSeries(row['geometry']).values[0]
            update_geo_series = affinity.scale(this_geo_series, xfact=1, yfact=y_stretch_factor, origin=(center_x, center_y))
            update_geo_series = round_geometry_coordinates(update_geo_series)
            #print(x_delta, y_delta, x_delta*y_delta, this_geo_series, '>>>', update_geo_series)
            #print(type(row['geometry']), type(update_geo_series))
            this_geometry = update_geo_series

            xmin, ymin, xmax, ymax = gpd.GeoSeries(update_geo_series).values[0].bounds
            x_delta = xmax - xmin
            y_delta = ymax - ymin
        
        if True:
            # Always dilate legend items for polygon
            center_x = xmin + x_delta*0.5
            center_y = ymin + y_delta*0.5

            update_geo_series = affinity.scale(this_geometry, xfact=1.1, yfact=1.1, origin=(center_x, center_y))
            update_geo_series = round_geometry_coordinates(update_geo_series)
            #print(x_delta, y_delta, x_delta*y_delta, this_geo_series, '>>>', update_geo_series)
            #print(type(row['geometry']), type(update_geo_series))
            this_geometry = update_geo_series

            xmin, ymin, xmax, ymax = gpd.GeoSeries(update_geo_series).values[0].bounds
            x_delta = xmax - xmin
            y_delta = ymax - ymin


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
                                            'geometry' : this_geometry
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


    bounding_box_height2 = []
    for index, row in layer2.iterrows():
        xmin, ymin, xmax, ymax = gpd.GeoSeries(row['geometry']).values[0].bounds
        x_delta = xmax - xmin
        y_delta = ymax - ymin
        if (x_delta*y_delta) > (rgb0.shape[0]*rgb0.shape[1])*0.8:
            continue
        bounding_box_height2.append(y_delta)

    bounding_box_height2 = np.array(bounding_box_height2)
    bounding_box_threshold2 = 20
    if bounding_box_height2.shape[0] > 0:
        bounding_box_threshold2 = min(np.quantile(bounding_box_height2, 0.5), np.mean(bounding_box_height2)) * 0.8
    #print(bounding_box_threshold2)

    for index, row in layer2.iterrows():
        xmin, ymin, xmax, ymax = gpd.GeoSeries(row['geometry']).values[0].bounds
        x_delta = xmax - xmin
        y_delta = ymax - ymin
        if (x_delta*y_delta) > (rgb0.shape[0]*rgb0.shape[1])*0.8:
            continue
        #print(x_delta, y_delta, x_delta*y_delta)

        this_geometry = row['geometry']
        # Stretch flat bounding box
        if y_delta < bounding_box_threshold2 and y_delta > 0:
            center_x = xmin + x_delta*0.5
            center_y = ymin + y_delta*0.5

            y_stretch_factor = max(1.2, bounding_box_threshold2 * 0.5 / y_delta)

            this_geo_series = gpd.GeoSeries(row['geometry']).values[0]
            update_geo_series = affinity.scale(this_geo_series, xfact=1, yfact=y_stretch_factor, origin=(center_x, center_y))
            update_geo_series = round_geometry_coordinates(update_geo_series)
            #print(x_delta, y_delta, x_delta*y_delta, this_geo_series, '>>>', update_geo_series)
            #print(type(row['geometry']), type(update_geo_series))
            this_geometry = update_geo_series

        updated_record = gpd.GeoDataFrame([{'name' : '', 
                                            'id' : int(legend_item_counter), 
                                            'description' : '', 
                                            'geometry' : this_geometry
                                            }])
        legend_item_counter += 1
        linked_ptln_description = gpd.GeoDataFrame(pd.concat( [linked_ptln_description, updated_record], ignore_index=True), crs=layer1.crs)

    linked_ptln_description1 = linked_ptln_description.set_crs('epsg:3857', allow_override=True)
    linked_ptln_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_PointLineType.geojson')), driver='GeoJSON')

    for index, row in linked_ptln_description1.iterrows():
        linked_ptln_description1.loc[linked_ptln_description1['id'] == row['id'], 'geometry'] = shapely.wkt.loads(str(row['geometry']).replace(', ', 'p').replace(' ', ' -').replace('p', ', ').replace('POLYGON -', 'POLYGON '))
    linked_ptln_description1.to_file(os.path.join(path_to_intermediate, 'intermediate7', target_map_name.replace('.tif', '_PointLineType_2.geojson')), driver='GeoJSON')


def round_geometry_coordinates(geom):
    if geom.geom_type == 'Polygon':
        exterior = [(round(x), round(y)) for x, y in geom.exterior.coords]
        interiors = [[(round(x), round(y)) for x, y in interior.coords] for interior in geom.interiors]
        return Polygon(exterior, interiors)
    else:
        # Handle other geometry types as needed
        return geom


def generating_json(target_map_name, input_image, output_dir, path_to_intermediate):
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

            original_file = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType_2.geojson')), driver='GeoJSON')
            for index, poi in original_file.iterrows():
                geo_series = gpd.GeoSeries(poi['geometry'])
                original_file.loc[index, 'geometry'] = geo_series.affine_transform(trans_matrix).values[0]
            original_file = original_file.set_crs('epsg:'+str(this_epsg_code), allow_override=True)
            original_file.to_file(os.path.join(output_dir, map_name.replace('.tif', '_PolygonType_crs.geojson')), driver='GeoJSON')


            original_file = gpd.read_file(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType_2.geojson')), driver='GeoJSON')
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

    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PolygonType_2.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PolygonType_qgis.geojson')))
    shutil.copyfile(os.path.join(path_to_intermediate, 'intermediate7', map_name.replace('.tif', '_PointLineType_2.geojson')), os.path.join(output_dir, map_name.replace('.tif', '_PointLineType_qgis.geojson')))

    print('================================================================================================================')
    print('Output at ' + str(output_dir)+' as json and geojson files...')
    print(os.path.join(output_dir, map_name.replace('.tif', '_PolygonType.geojson')), 'for legend item segmentation (polygon) (GPKG_GEOJSON format, image coordinate)')
    print(os.path.join(output_dir, map_name.replace('.tif', '_PointLineType.geojson')), 'for legend item segmentation (point, line) (GPKG_GEOJSON format, image coordinate)')
    print(os.path.join(output_dir, map_name.replace('.tif', '_[Type]_internal.json')), 'for internal usage (competition format)')
    if postprocessing_for_crs == True and crs_flag == True:
        print(os.path.join(output_dir, map_name.replace('.tif', '_[Type]_crs.geojson')), 'for output with transformed crs (map coordinate)')
    print(os.path.join(output_dir, map_name.replace('.tif', '_[Type]_qgis.geojson')), 'for output visualizable in qgis (qgis coordinate)')
    print('Legend-item Segmentation has concluded for input image:', input_image)
    print('================================================================================================================')

    return True





def start_linking_postprocessing(target_map_name, input_image, output_dir, path_to_intermediate, input_area_segmentation, input_legend_segmentation, preprocessing_for_cropping, postprocessing_for_crs, competition_custom):
    if '.tif' not in target_map_name:
        target_map_name = target_map_name + '.tif'
    
    print('Step ( 8/10): Preparing output json - Generating GEOJSON file (GPKG schema)...')
    reading_raster_output(target_map_name, path_to_intermediate)
    compiling_geojson(target_map_name, input_image, output_dir, path_to_intermediate)
    print('Step ( 9/10): Preparing output json - Generating JSON file (competition format)...')
    generating_json(target_map_name, input_image, output_dir, path_to_intermediate)
    print('Step (10/10): Finalizing legend-item segmentation...')
    adjusting_crs(target_map_name, input_image, path_to_intermediate, output_dir, postprocessing_for_crs)




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
            path_to_intermediate = 'Example_Output/LINK_Intermediate/testing/'+str(target_map)+'/'
            input_legend_segmentation = 'Uncharted/ch2_validation_evaluation_labels_coco.json'

            os.makedirs(os.path.dirname(output_dir), exist_ok=True)

            start_linking_postprocessing(target_map_name, input_image, output_dir, path_to_intermediate, None, input_legend_segmentation, False, True, True)
            this_map_count += 1
            print('Processed map... '+str(target_map)+'.tif'+'   ...'+str(this_map_count)+'/'+str(total_map_count))


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main()