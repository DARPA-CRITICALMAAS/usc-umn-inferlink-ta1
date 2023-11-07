from osgeo import ogr, osr
import os
from osgeo import gdal
from osgeo.gdalconst import *
from shapely.ops import linemerge
from shapely.geometry import LineString
from shapely.wkt import loads
import math
from dash_pattern_direction import detect_line_dash_direction
from legend_detection import detect_legend
from helper.process_shp import write_shp_in_imgcoord, rm_dup_lines, integrate_lines, write_shp_in_imgcoord_output_schema

def write_shp_in_imgcoord_output_schema(shp_name, all_lines, coor_path, legend_text = None, image_coords=False):
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Deleting the exit shapefile
    if os.path.exists(shp_name):
        driver.DeleteDataSource(shp_name)
        
    
    srs = gdal.Open(coor_path).GetProjection()
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(srs)
#     spatial_reference.ImportFromProj4(srs)
    # Creating the shapefile
    ds = driver.CreateDataSource(shp_name)
    layer = ds.CreateLayer('layerName', spatial_reference, geom_type = ogr.wkbLineString)
    
    if ds is None:
        print ('Could not create file')
        sys.exit(1)
    
    fieldDefn = ogr.FieldDefn('ID', ogr.OFTInteger)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('geometr', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('name', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('direction', ogr.OFTInteger)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('type', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('descript', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('dash', ogr.OFTString)
    layer.CreateField(fieldDefn)
    fieldDefn = ogr.FieldDefn('symbol', ogr.OFTString)
    layer.CreateField(fieldDefn)
    
    transform = gdal.Open(coor_path).GetGeoTransform()
    
    cnt = 0
    
    for line_wkt, attr in all_lines.items():
        cnt += 1
        line = loads(line_wkt)
        lineString = ogr.Geometry(ogr.wkbLineString)
        if isinstance(line, list):
            for v in line:
                if not image_coords:
                    geo_x = transform[0] + v[1] * transform[1] + v[0] * transform[2]
                    geo_y = transform[3] + v[1] * transform[4] + v[0] * transform[5]
                    lineString.AddPoint(geo_x, geo_y)
                else:
#                     print('--- write in image coordinate ---')
                    lineString.AddPoint(v[1], -v[0])
        else:
            for p in list(line.coords):
                if not image_coords:
                    geo_x = transform[0] + p[1] * transform[1] + p[0] * transform[2]
                    geo_y = transform[3] + p[1] * transform[4] + p[0] * transform[5]
                    lineString.AddPoint(geo_x, geo_y)
                else:
#                     print('--- write in image coordinate ---')
                    lineString.AddPoint(p[1], -p[0])

        featureDefn = layer.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(lineString)

        feature.SetField('ID', cnt)
        feature.SetField('geometr', 'line')
        feature.SetField('name', 'fault line')
        feature.SetField('direction', attr[1])
        feature.SetField('type', None)
        feature.SetField('descript', legend_text)
        feature.SetField('dash', attr[0])
        if attr[1] != 0:
            feature.SetField('symbol', 'ball-and-bar symbol')
        else:
            feature.SetField('symbol', None)

        layer.CreateFeature(feature)
        lineString.Destroy()
        feature.Destroy()
    ds.Destroy()
    print ("Shapefile created")    
        
if __name__ == '__main__':
    map_list = ['NV_HiddenHills']

    for map_name in map_list:
        shapefile_path = f'./pred4shp/{map_name}_fault_line_pred.shp'
        out_shapefile_path = f'./hackathon1/{map_name}_fault_line_image.shp'
    #     out_geopackage = out_shapefile_path[:-4] + '.gpkg'
        out_geojson = out_shapefile_path[:-4] + '.geojson'
        map_image_path = f'/data/weiweidu/criticalmaas_data/validation_fault_line_comb/{map_name}.png'
#         tif_image_path = f'/data/weiweidu/criticalmaas_data/validation/{map_name}.tif'
        tif_image_path = f'/data/weiweidu/criticalmaas_data/hackathon1/output_georeferencing/{map_name}.geotiff.tif'
        tif_dir = '/data/weiweidu/criticalmaas_data/validation' # for json file
        map_dir = '/data/weiweidu/criticalmaas_data/validation_fault_line_comb'

        line_dict = detect_line_dash_direction(shapefile_path, map_image_path, tif_dir, \
                                              obj_name='normal_fault_line', match_threshold=0.5)
    #################################  
    # refine (remove duplicates) lines
    #     refined_line_dict = {}
    #     for dash_type, lines in line_dict.items():
    #         if lines == []:
    #             continue
    #         nodup_lines = rm_dup_lines(lines)
    #         merged_lines = integrate_lines(nodup_lines)
    #         refined_line_dict[dash_type] = merged_lines
    #     write_shp_in_imgcoord_output_schema('./pred4shp/OR_Carlton_fault_line_pred_dash.shp', line_dict, tif_image_path)
    #################################  

        legend = detect_legend(map_name, map_dir, tif_dir)
        legend_str = '. '.join(legend)

        write_shp_in_imgcoord_output_schema(out_shapefile_path, line_dict, tif_image_path, legend_str, image_coords=True)

        import geopandas
        shp_file = geopandas.read_file(out_shapefile_path)
        shp_file.to_file(out_geojson, driver='GeoJSON')
        print('*** save the predicted geojson in {} ***'.format(out_geojson))
    
#     import geopandas as gpd
    
#     gdf = gpd.read_file(out_shapefile_path)
#     gdf.to_file(out_geopackage, driver='GPKG')
#     print('--- saved geopackage in {} ---'.format(out_geopackage))
    