from osgeo import ogr, osr
import os
from osgeo import gdal
from osgeo.gdalconst import *
from shapely.ops import linemerge
from shapely.geometry import LineString
from shapely.wkt import loads
import math

def reverse_geom_coords(geom):
    if isinstance(geom, LineString):
        return LineString([(x, -y) for x, y in geom.coords])
    elif isinstance(geom, (MultiLineString)):
        multilines = []
        for i, ln in enumerate(geom.geoms):
            multilines.append(LineString([(x, -y) for x, y in ln.coords]))
        return MultiLineString(multilines)
    else:
        raise ValueError(f"Geometry type '{type(geom)}' not supported")

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
    
    for line_cat, lines in all_lines.items():
        for line in lines:
#             line = reverse_geom_coords(line)
            cnt += 1
            lineString = ogr.CreateGeometryFromWkb(line.wkb)
            featureDefn = layer.GetLayerDefn()
            feature = ogr.Feature(featureDefn)
            feature.SetGeometry(lineString)

            feature.SetField('ID', cnt)
            feature.SetField('geometr', 'line')
            feature.SetField('name', 'fault line')
            feature.SetField('direction', None)
            feature.SetField('type', None)
            feature.SetField('descript', legend_text)
            feature.SetField('dash', line_cat)


            layer.CreateFeature(feature)
            lineString.Destroy()
            feature.Destroy()
    ds.Destroy()
    print ("Shapefile created")    
    
        
if __name__ == '__main__':
    map_name = 'NV_HiddenHills'
    in_shapefile_path = '/data/weiweidu/LDTR_criticalmaas_test/pred4shp/NV_HiddenHills_fault_line_pred.shp'
    out_shapefile_path = '/data/weiweidu/line_github_workspace/pred_maps/NV_HiddenHills_fault_line_pred_attr.shp'
    input_patch_dir = '/data/weiweidu/LDTR_criticalmaas/data/darpa/fault_lines/NV_HiddenHills_g256_s100/raw'
    #'/data/weiweidu/line_github_workspace/gpt4_outputs'

#     line_dict = extract_attributes_along_line(map_name, in_shapefile_path, input_patch_dir)
#     write_shp_in_imgcoord_with_attr(out_shapefile_path, line_dict, image_coords=True)

    