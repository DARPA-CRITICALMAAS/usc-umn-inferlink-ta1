import numpy as np
import rasterio
import pyproj
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, MultiPoint, MultiLineString, Polygon, LineString
from shapely.geometry import shape, mapping
from shapely.affinity import affine_transform
import json

def reverse_geom_coords(geom):
    """
    Reverses the coordinates (x, y) to (y, x) for the input geometry.
    Supports Points, LineStrings, Polygons, and their Multi* counterparts.
    """
    if geom.is_empty:
        return geom
    if isinstance(geom, (Polygon)):
        return Polygon([(x, y) for x, y in geom.exterior.coords])
    elif isinstance(geom, (MultiPolygon)):
        multipoly = []
        for i, poly in enumerate(geom.geoms):
            multipoly.append(Polygon([(x, y) for x, y in geom.exterior.coords]))
        return MultiPolygon(multipoly)
    elif isinstance(geom, (Point, MultiPoint)):
        return Point(geom.x, geom.y)
    elif isinstance(geom, (LineString)):
        return LineString([(y, x) for x, y in geom.coords])
    elif isinstance(geom, (MultiLineString)):
        multiline = []
        for i, line in enumerate(geom.geoms):
            multiline.append(LineString([(x, y) for x, y in line.coords]))
        return MultiLineString(multiline)
    else:
        # Extend to other geometry types if necessary
        raise ValueError(f"Geometry type '{type(geom)}' not supported")

def img2geo_geojson_geotif(geotiff_file, input_geojson):
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

def gcps2transform_matrix(gcps):
    """
    GCPs: [(x_src, y_src, x_dst, y_dst), ...]
    """
    gcps = gcps[:3]
    src_pts = np.array([(pt[0], pt[1], 1) for pt in gcps])#[:3,:]
    dst_pts = np.array([(pt[2], pt[3]) for pt in gcps])#[:3,:]

    # Preparing the matrices for solving
    A = np.vstack([src_pts.T[:2], np.ones(len(gcps))]).T
    B = dst_pts

    # Solving for the transformation matrix
    transformation_matrix, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    
    return transformation_matrix

def apply_affine_transformation(geom, matrix):
    # Convert GeoJSON geometry to a Shapely geometry
    shapely_geom = shape(geom)
    # Apply the affine transformation
    transformed_geom = affine_transform(shapely_geom, [matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1], matrix[0,2], matrix[1,2]])
    # Convert back to GeoJSON geometry
    return mapping(transformed_geom)

def img2geo_geojson_gcp(transform_matrix, input_geojson):
    with open(input_geojson, 'r') as f:
        geojson_data = json.load(f)
    # Apply the transformation to each feature
    for feature in geojson_data['features']:
        geometry = shape(feature['geometry'])
#         transformed_geometry = affine_transform(geometry, transform_matrix)
#         # Update the geometry in the original GeoJSON feature
        feature['geometry'] = reverse_geom_coords(geometry)
        feature['geometry'] = apply_affine_transformation(feature['geometry'], transform_matrix.T)
    
    f_name = input_geojson.split('/')[-1].split('.')[0]
    output_geojson_path = f"./temp/{f_name}_georef.geojson"
    with open(output_geojson_path, 'w') as f:
        json.dump(geojson_data, f)
    return output_geojson_path

        
    