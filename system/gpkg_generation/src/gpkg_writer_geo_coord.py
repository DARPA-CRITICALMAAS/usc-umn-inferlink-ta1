import os, sys
import time
import json
from gpkg_helper import img2geo_geojson_gcp as img2geo_geojson #img2geo_geometry
from gpkg_helper import gcps2transform_matrix
from criticalmaas.ta1_geopackage import GeopackageDatabase
import concurrent.futures
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, MultiLineString, Polygon, LineString
from pytest import mark
import rasterio
import logging
import sqlalchemy 

logger = logging.getLogger(__name__)

poly_feat_id, poly_type_id, geo_unit_id, ln_type_id, ln_feat_id, pt_feat_id, pt_type_id = 0, 0, 0, 0, 0, 0, 0

def get_feature_from_geojson(input_path):
    with open(input_path, 'r') as input_file:
        feat_json = json.load(input_file)
    return feat_json['features']

def read_georef_output(input_path):
    with open(input_path, 'r') as input_file:
        gcp_dict = json.load(input_file)
    return gcp_dict[0]['map']['projection_info']

def create_gpkg(output_path, legend_json, map_name, crs):
    db = GeopackageDatabase(
      output_path,
      crs=crs # Geographic coordinates (default)
      # crs="CRITICALMAAS:pixel" # Pixel coordinates
    )
    
    # Insert types (required for foreign key constraints)
    img_height, img_width = legend_json['map_dimension']
    db.write_models([
      db.model.map(id=map_name, name=map_name, image_url="test", source_url='test', image_width =img_width, image_height=img_height),
    ])
    return db


def write_georef_into_gpkg(db, georef_json, map_name):   
    
    projection = georef_json['projection']
    
    geo_gcps = []
    gcp_feat_list  = []
    gcp_pairs = []
    
    for item in georef_json['gcps']:
        geo_gcps.append(item['map_geom'])
        gcp_pairs.append(item['px_geom'] + item['map_geom'])

        gcp_feat = {"geometry": {
                            "type": "Point",
                            "coordinates": item['map_geom'],
                            },
                         "properties": {
                            "id":item['id'],
                            "map_id": map_name,
                            "x": item['px_geom'][0],
                            "y": item['px_geom'][1],
                            "confidence": item['confidence'],
                            "provenance": item['provenance'],

                            }
                        }
        gcp_feat_list.append(gcp_feat)
    
    georef_meta_feat = {"geometry": {
                                    "type": "Polygon",
                                    "coordinates": [geo_gcps + [geo_gcps[0]]],
                                    },
                         "properties": {
                            "id":map_name,
                            "map_id": map_name,
                            "projection": projection,
                            "provenance": 'modelled',
                            }
                    }
    
    db.write_features("georeference_meta", [georef_meta_feat])
    
    db.write_features("ground_control_point", gcp_feat_list)
    
    return gcp_pairs

@mark.parametrize("engine", ["fiona", "pyogrio"])
def write_poly_into_gpkg(db, feat_list, map_name, crs):     
    engine = 'pyogrio' 
    global poly_feat_id, poly_type_id, geo_unit_id
    geo_unit_list = []
    poly_type_list = []
    poly_geom, poly_property = [], []
    
    if len(feat_list) == 0:
        return
    
    for ind, feat in enumerate(feat_list):        
        geo_poly = feat['geometry']['coordinates']
        multi_geo_poly = MultiPolygon([Polygon(i) for i in geo_poly])
#         print(multi_geo_poly)
        if len(geo_poly[0]) == 0: # skip empty geom
            continue
        
        poly_feat = {"id":str(poly_feat_id),
                    "map_id": map_name,
                    "type": str(poly_type_id),
                    "confidence": None,
                    "provenance": 'modelled',
                    "geometry": multi_geo_poly #MultiPolygon(multi_geo_poly),
                }
        poly_geom.append(poly_feat)
        poly_feat_id += 1
        
    geo_unit_feat = {'id': str(geo_unit_id), 'name':feat['properties']['GeologicUnit']['name'],'description':feat['properties']['GeologicUnit']['description'], 'age_text':feat['properties']['GeologicUnit']['age_text'],'t_interval':feat['properties']['GeologicUnit']['t_interval'], 'b_interval':feat['properties']['GeologicUnit']['b_interval'], 't_age': feat['properties']['GeologicUnit']['t_age'] if feat['properties']['GeologicUnit']['t_age'] and feat['properties']['GeologicUnit']['t_age'] else None, 'b_age':feat['properties']['GeologicUnit']['b_age'] if feat['properties']['GeologicUnit']['b_age'] and feat['properties']['GeologicUnit']['b_age'] else None,'lithology':feat['properties']['GeologicUnit']['lithology']}

    geo_unit_list.append(geo_unit_feat)
    

    poly_type_feat = {'id': str(poly_type_id), 'name': "geologic unit", 'description': feat['properties']['PolygonType']['description'], 'color': feat['properties']['PolygonType']['color'] if feat['properties']['PolygonType']['color'] else 'not detected', 'pattern': feat['properties']['PolygonType']['pattern'], 'abbreviation': feat['properties']['PolygonType']['abbreviation'], 'category': feat['properties']['PolygonType']['category'], 'map_unit': str(geo_unit_id)}

    poly_type_list.append(poly_type_feat)
    poly_type_id += 1
    geo_unit_id += 1
#     print(poly_type_id, geo_unit_id)    
    df_geo_unit = pd.DataFrame(geo_unit_list)
    df_geo_unit.to_sql('geologic_unit', db.engine , if_exists='append', index=False)
    
    df_poly_type = pd.DataFrame(poly_type_list)
    df_poly_type.to_sql('polygon_type', db.engine , if_exists='append', index=False)
    
    df_poly_geom = pd.DataFrame(poly_geom)
    gdf = gpd.GeoDataFrame(df_poly_geom, crs=crs)
    gdf.to_file(
        db.file,
        layer="polygon_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )

        
def write_pt_into_gpkg(db, feat_list, map_name, crs):   
    global pt_type_id, pt_feat_id
    engine = 'pyogrio'
    pt_type_list = []
    pt_feat_list = []
    
    if len(feat_list) == 0:
        return
    
    for ind, feat in enumerate(feat_list):
        geo_pt = Point(feat['geometry']['coordinates'])
        
        if len(feat['geometry']['coordinates']) == 0:
            continue # skip the empty geom
        
        pt_name = ' '.join(feat['properties']['type'].split('_'))
        pt_type = {'id':str(pt_type_id), 'name':pt_name, 'description':feat['properties']['type']}
        
        if pt_type not in pt_type_list:
            pt_type_list.append(pt_type)
            cur_pt_type_id = len(pt_type_list) - 1
        else:
            cur_pt_type_id = pt_type_list.index(pt_type)
        
        pt_feat = {"id":str(pt_feat_id),
                    "map_id": map_name,
                    "type": str(pt_type_id),
                    "dip_direction": 0,
                    "dip": 0,  
                    "confidence": None,
                    "provenance": 'modelled',
                    "geometry": geo_pt#Point(geo_pt)
                    }
        pt_feat_list.append(pt_feat)
        pt_feat_id += 1
    
    
    if len(pt_feat_list) == 0:
        return 
    
    for i in range(len(pt_type_list)):
        pt_type_list[i]['id'] = str(pt_type_id + i)
    
    pt_type_id += len(pt_type_list)
    
    df_pt_type = pd.DataFrame(pt_type_list)
    df_pt_type.to_sql('point_type', db.engine , if_exists='append', index=False)
    
    df_pt_geom = pd.DataFrame(pt_feat_list)
    gdf = gpd.GeoDataFrame(df_pt_geom, crs=crs)
    gdf.to_file(
        db.file,
        layer="point_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    
    return    
        
def write_ln_into_gpkg(db, feat_list, map_name, crs): 
    global ln_type_id, ln_feat_id
    engine = 'pyogrio'
    ln_type_list, ln_feat_list = [], []
    
    if len(feat_list) == 0:
        return
    
    for ind, feat in enumerate(feat_list):
#         print(feat['geometry']['coordinates'])
        if feat['geometry']['type'] == 'MultiLineString':
            ln_geom = MultiLineString(feat['geometry']['coordinates'])
        else:
            ln_geom = LineString(feat['geometry']['coordinates'])
        if len(feat['geometry']['coordinates']) == 0: # skip the empty geom
            continue
        
        ln_type = {'name':feat['properties']['name'][:-5], 'description':feat['properties']['descript'], \
                   'dash_pattern':feat['properties']['dash'], 'symbol':feat['properties']['symbol']}
        
        if ln_type not in ln_type_list:
            ln_type_list.append(ln_type)
            cur_ln_type_id = len(ln_type_list) - 1
        else:
            cur_ln_type_id = ln_type_list.index(ln_type)
        
        ln_feat = {
                    "id":str(ln_feat_id),
                    "map_id": map_name,
                    "name": feat['properties']['name'],
                    "type": str(cur_ln_type_id),
                    "polarity": 0, 
                    "confidence": None,
                    "provenance": 'modelled',
                    "geometry": ln_geom
                    }
        ln_feat_list.append(ln_feat)  

        ln_feat_id += 1
        
    
    if len(ln_feat_list) == 0:
        return
    
    for i in range(len(ln_type_list)):
        ln_type_list[i]['id'] = str(ln_type_id + i)
    
    ln_type_id += len(ln_type_list)
    
    df_ln_type = pd.DataFrame(ln_type_list)
    df_ln_type.to_sql('line_type', db.engine , if_exists='append', index=False)
    
    df_ln_geom = pd.DataFrame(ln_feat_list)
    gdf = gpd.GeoDataFrame(df_ln_geom, crs=crs)
    gdf.to_file(
        db.file,
        layer="line_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    return
    
def write_gpkg(output_dir, map_name, layout_output_path, \
               georef_output_path, poly_output_path, ln_output_path, pt_output_path, logger):    
    
    with open(layout_output_path, 'r') as json_file:
        legend_item_descr_json = json.load(json_file)
        
    out_gpkg_path = f'{output_dir}/{map_name}_geo.gpkg'    
    
    # read ground control points
    georef_json = read_georef_output(georef_output_path) 
    crs = georef_json.get('projection', 'EPSG:4326')
    
    try:
        db_instance = create_gpkg(out_gpkg_path, legend_item_descr_json, map_name,  crs)
        logger.info(f'Created a gpkg in the geo-coordinate')
    except sqlalchemy.exc.IntegrityError as error:
        logger.error(f'{out_gpkg_path} exists. Please delete it.')
        sys.exit(1)
    
    gcps = write_georef_into_gpkg(db_instance, georef_json, map_name)
    trans_matrix = gcps2transform_matrix(gcps)

    # write polygon features    
    if os.path.exists(poly_output_path):
        poly_files = os.listdir(poly_output_path)
        for i, poly_geojson in enumerate(poly_files):
            if '.geojson' not in poly_geojson or '_empty' in poly_geojson:
                continue                        
            img_poly_geojson_path = os.path.join(poly_output_path, poly_geojson)
            geo_poly_geojson_path = img2geo_geojson(trans_matrix, img_poly_geojson_path)
            features = get_feature_from_geojson(geo_poly_geojson_path)    
            logger.info(f'Writing {i+1}/{len(poly_files)} polygon GeoJson into the geo GPKG')
            write_poly_into_gpkg(db_instance, features[:], map_name, crs)
    logger.info(f'All polygon GeoJson is written into the geo GPKG')
    
    # write point features
    if os.path.exists(pt_output_path):
        pt_files = os.listdir(pt_output_path)
        for i, pt_geojson in enumerate(pt_files):
            if '.geojson' not in pt_geojson or '_empty' in pt_geojson:
                continue  
            img_pt_geojson_path = os.path.join(pt_output_path, pt_geojson)
            geo_pt_geojson_path = img2geo_geojson(trans_matrix, img_pt_geojson_path)
            features = get_feature_from_geojson(geo_pt_geojson_path)    
            logger.info(f'Writing {i+1}/{len(pt_files)} point GeoJson into the geo GPKG')
            write_pt_into_gpkg(db_instance, features[:], map_name, crs)
    logger.info(f'All point GeoJson is written into the geo GPKG')
    # write line features
    if os.path.exists(ln_output_path):
        ln_files = os.listdir(ln_output_path)
        for i, ln_geojson in enumerate(ln_files):
            if '.geojson' not in ln_geojson or '_empty' in ln_geojson:
                continue
            img_ln_geojson_path = os.path.join(ln_output_path, ln_geojson)
            geo_ln_geojson_path = img2geo_geojson(trans_matrix, img_ln_geojson_path)
            features = get_feature_from_geojson(geo_ln_geojson_path)
            logger.info(f'Writing {i+1}/{len(ln_files)} line GeoJson into the geo GPKG')
            write_ln_into_gpkg(db_instance, features[:], map_name, crs)
    logger.info(f'All line GeoJson is written into the GPKG')
# if __name__ == '__main__':
#     write_gpkg(output_dir, map_name, layout_output_path, \
#                georef_output_path, poly_output_path, ln_output_path, pt_output_path)
        