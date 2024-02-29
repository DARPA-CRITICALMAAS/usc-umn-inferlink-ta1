import numpy as np
import rasterio
import pyproj
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, MultiPoint, MultiLineString, Polygon, LineString

def reverse_geom_coords(geom):
    """
    Reverses the coordinates (x, y) to (y, x) for the input geometry.
    Supports Points, LineStrings, Polygons, and their Multi* counterparts.
    """
    if geom.is_empty:
        return geom
    if isinstance(geom, (Polygon, MultiPolygon)):
        return Polygon([(y, x) for x, y in geom.exterior.coords])
    elif isinstance(geom, (Point, MultiPoint)):
        return Point(geom.y, geom.x)
    elif isinstance(geom, (LineString, MultiLineString)):
        return LineString([(y, x) for x, y in geom.coords])
    else:
        # Extend to other geometry types if necessary
        raise ValueError(f"Geometry type '{type(geom)}' not supported")

def img2geo_geojson(geotiff_file, input_geojson):
    # convert the image to a binary raster .tif
    raster = rasterio.open(geotiff_file)
    transform = raster.transform
    array     = raster.read(1)
    crs       = raster.crs
    width     = raster.width
    height    = raster.height
    raster.close()
#     this_epsg_code = pyproj.crs.CRS.from_proj4(crs.to_proj4()).to_epsg()
    trans_np = np.array(transform) 
    print(trans_np)
#     trans_matrix = [trans_np[0], trans_np[1], trans_np[3], trans_np[4], trans_np[2], trans_np[5]]
    trans_matrix = [trans_np[0], trans_np[1], trans_np[3], trans_np[4], trans_np[2], trans_np[5]]

    original_file = gpd.read_file((input_geojson), driver='GeoJSON')
    for index, poi in original_file.iterrows():
        if isinstance(poi['geometry'], (LineString, MultiLineString)):
            geo_series = gpd.GeoSeries(poi['geometry']).apply(reverse_geom_coords)
        else:
            geo_series = gpd.GeoSeries(poi['geometry'])
        original_file.loc[index, 'geometry'] = geo_series.affine_transform(trans_matrix).values[0]
#     original_file = original_file.set_crs('epsg:'+str(this_epsg_code), allow_override=True)
    
    f_name = input_geojson.split('/')[-1].split('.')[0]
    output_geojson_path = f"./temp/{f_name}_georef.geojson"
    original_file.to_file((output_geojson_path), driver='GeoJSON')
    return output_geojson_path

def img2qgis_geometry(geometry, geom_type):
    """
    Convert a point from (row, col) to (col, -row) for qgis visualization. 
    :param geom_type: ['polyon', 'line', 'point']
    """
    if geom_type == 'polygon':
        geo_poly = [[pt[0], -pt[1]] for pt in geometry[0]]
        return [[geo_poly]]
    
    if geom_type == 'line':
        geo_line = [[pt[1], -pt[0]] for pt in geometry]
        return [geo_line]
    
    if geom_type == 'point':
        geo_point = [geometry[0], -geometry[1]]
        return geo_point
        
    