from osgeo import ogr, osr
import os
from osgeo import gdal
from osgeo.gdalconst import *
from shapely.ops import linemerge
from shapely.geometry import LineString
from shapely.wkt import loads
import math
from line_ornament import extract_attributes_along_line

def write_shp_in_imgcoord_with_attr(shp_name, all_lines, legend_text=None, feature_name=None, image_coords=False):
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    
  # Getting shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')
    # Deleting the exit shapefile
    if os.path.exists(shp_name):
        driver.DeleteDataSource(shp_name)
        
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(4326)

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
    
#     transform = gdal.Open(coor_path).GetGeoTransform()
    
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
        feature.SetField('direction', None)
        feature.SetField('type', None)
        feature.SetField('descript', legend_text)
        feature.SetField('dash', attr[0])
        if len(attr) == 2:
            feature.SetField('symbol', 'ball-and-bar symbol')
            feature.SetField('direction', attr[1])
        else:
            feature.SetField('symbol', None)

        layer.CreateFeature(feature)
        lineString.Destroy()
        feature.Destroy()
    ds.Destroy()
    print ("Shapefile created")    
    
    import geopandas
    geojson_path = f'{shp_name[:-4]}.geojson'
    shp_file = geopandas.read_file(shp_name)
    shp_file.to_file(geojson_path, driver='GeoJSON')
    print('*** save the predicted geojson in {} ***'.format(geojson_path))
        
if __name__ == '__main__':
    map_name = 'NV_HiddenHills'
    in_shapefile_path = '/data/weiweidu/LDTR_criticalmaas_test/pred4shp/NV_HiddenHills_fault_line_pred.shp'
    out_shapefile_path = '/data/weiweidu/line_github_workspace/pred_maps/NV_HiddenHills_fault_line_pred_attr.shp'
    input_patch_dir = '/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines/NV_HiddenHills_g256_s100/raw'
    #'/data/weiweidu/line_github_workspace/gpt4_outputs'

    line_dict = extract_attributes_along_line(map_name, in_shapefile_path, input_patch_dir)
    write_shp_in_imgcoord_with_attr(out_shapefile_path, line_dict, image_coords=True)

    