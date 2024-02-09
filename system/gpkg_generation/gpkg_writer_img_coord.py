import os
import time
import json
from gpkg_helper import img2qgis_geometry
from criticalmaas.ta1_geopackage import GeopackageDatabase
import concurrent.futures
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, MultiLineString
from pytest import mark

poly_id, ln_id, pt_id = 0, 0, 0

def get_feature_from_geojson(input_path):
    with open(input_path, 'r') as input_file:
        feat_json = json.load(input_file)
    return feat_json['features']


def create_gpkg(output_path, legend_json, map_name):
    db = GeopackageDatabase(
      output_path,
#       crs="EPSG:4326" # Geographic coordinates (default)
      crs="CRITICALMAAS:pixel" # Pixel coordinates
    )
    
    # Insert types (required for foreign key constraints)
    img_height, img_width = legend_json['map_dimension']

    db.write_models([
      db.model.map(id=map_name, name=map_name, image_url="test", source_url='test', image_width =img_width, image_height=img_height),
    ])
    return db


@mark.parametrize("engine", ["fiona", "pyogrio"])
def write_poly_into_gpkg(db, feat_list, map_name):     
    from sqlalchemy import create_engine
    engine = 'pyogrio' 
    global poly_id
    geo_unit_list = []
    poly_type_list = []
    poly_geom, poly_property = [], []
    
    for ind, feat in enumerate(feat_list):        
        geo_poly = img2qgis_geometry(feat['geometry']['coordinates'], 'polygon')
        
        if len(geo_poly[0]) == 0: # skip empty geom
            continue
        
        poly_feat = {"id":str(poly_id),
                    "map_id": map_name,
                    "type": str(poly_id),
                    "confidence": None,
                    "provenance": 'modelled',
                    "geometry": MultiPolygon(geo_poly),
                }
        poly_geom.append(poly_feat)
        
        geo_unit_feat = {'id': str(poly_id), 'name':feat['properties']['GeologicUnit']['name'],'description':feat['properties']['GeologicUnit']['description'], 'age_text':feat['properties']['GeologicUnit']['age_text'],'t_interval':feat['properties']['GeologicUnit']['t_interval'], 'b_interval':feat['properties']['GeologicUnit']['b_interval'], 't_age':10 if isinstance(feat['properties']['GeologicUnit']['t_age'], str) else feat['properties']['GeologicUnit']['t_age'], 'b_age':10 if isinstance(feat['properties']['GeologicUnit']['b_age'], str) else feat['properties']['GeologicUnit']['b_age'],'lithology':feat['properties']['GeologicUnit']['lithology']}
        
        geo_unit_list.append(geo_unit_feat)
        
        poly_type_feat = {'id': str(poly_id), 'name': "geologic unit", 'description': feat['properties']['PolygonType']['description'], 'color': feat['properties']['PolygonType']['color'] if feat['properties']['PolygonType']['color'] else 'not detected', 'pattern': feat['properties']['PolygonType']['pattern'], 'abbreviation': feat['properties']['PolygonType']['abbreviation'], 'category': feat['properties']['PolygonType']['category'], 'map_unit': str(poly_id)}
        
        poly_type_list.append(poly_type_feat)
        
        poly_id += 1
        
    if len(poly_type_list)  == 0:
        return 
    
    df_geo_unit = pd.DataFrame(geo_unit_list)
    df_geo_unit.to_sql('geologic_unit', db.engine , if_exists='append', index=False)
    
    df_poly_type = pd.DataFrame(poly_type_list)
    df_poly_type.to_sql('polygon_type', db.engine , if_exists='append', index=False)
    
    df_poly_geom = pd.DataFrame(poly_geom)
    gdf = gpd.GeoDataFrame(df_poly_geom, crs="EPSG:4326")
    gdf.to_file(
        db.file,
        layer="polygon_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    return

        
def write_pt_into_gpkg(db, feat_list, map_name):   
    global pt_id
    engine = 'pyogrio'
    pt_type_list = []
    pt_feat_list = []
    for ind, feat in enumerate(feat_list):
        geo_pt = img2qgis_geometry(feat['geometry']['coordinates'], 'point')
        
        if len(geo_pt) == 0:
            continue # skip the empty geom
        
        pt_feat = {"id":str(pt_id),
                    "map_id": map_name,
                    "type": str(pt_id),
                    "dip_direction": 0,
                    "dip": 0,  
                    "confidence": None,
                    "provenance": 'modelled',
                    "geometry": Point(geo_pt)
                    }
        pt_feat_list.append(pt_feat)
        
        pt_type = {'id':str(pt_id), 'name':"other", 'description':None}
        pt_type_list.append(pt_type)
        
        pt_id += 1
    
    if len(pt_feat_list) == 0:
        return
    
    df_pt_type = pd.DataFrame(pt_type_list)
    df_pt_type.to_sql('point_type', db.engine , if_exists='append', index=False)
    
    df_pt_geom = pd.DataFrame(pt_feat_list)
    gdf = gpd.GeoDataFrame(df_pt_geom, crs="EPSG:4326")
    gdf.to_file(
        db.file,
        layer="point_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    return
        
        
def write_ln_into_gpkg(db, feat_list, map_name): 
    global ln_id
    engine = 'pyogrio'
    ln_type_list, ln_feat_list = [], []
    for ind, feat in enumerate(feat_list):
        ln_geom = img2qgis_geometry(feat['geometry']['coordinates'], 'line')
        
        if len(ln_geom[0]) == 0: # skip the empty geom
            continue
        
        ln_feat = {
                    "id":str(ln_id),
                    "map_id": map_name,
                    "name": feat['properties']['name'],
                    "type": str(ln_id),
                    "polarity": 0, 
                    "confidence": None,
                    "provenance": 'modelled',
                    "geometry": MultiLineString(ln_geom)
                    }
        ln_feat_list.append(ln_feat)            
        
        ln_type = {'id':str(ln_id), 'name':"fault", 'description':feat['properties']['descript'], \
                   'dash_pattern':feat['properties']['dash'], 'symbol':feat['properties']['symbol']}
        ln_type_list.append(ln_type)
        
        ln_id += 1
    
    if len(ln_feat_list) == 0:
        return 
    df_ln_type = pd.DataFrame(ln_type_list)
    df_ln_type.to_sql('line_type', db.engine , if_exists='append', index=False)
    
    df_ln_geom = pd.DataFrame(ln_feat_list)
    gdf = gpd.GeoDataFrame(df_ln_geom, crs="EPSG:4326")
    gdf.to_file(
        db.file,
        layer="line_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    return

    
def write_gpkg(output_dir, map_name, \
               layout_output_path, georef_output_path, poly_output_path, ln_output_path, pt_output_path):    
    
    with open(layout_output_path, 'r') as json_file:
        legend_item_descr_json = json.load(json_file)
        
    out_gpkg_path = f'{output_dir}/{map_name}.gpkg'    
    db_instance = create_gpkg(out_gpkg_path, legend_item_descr_json, map_name)
    
 # write polygon features    
    if os.path.exists(poly_output_path):
        poly_files = os.listdir(poly_output_path)
        for poly_geojson in poly_files:
            if '.geojson' not in poly_geojson:
                continue
            geojson_path = os.path.join(poly_output_path, poly_geojson)
            features = get_feature_from_geojson(geojson_path)    
            write_poly_into_gpkg(db_instance, features[:], map_name)
                    
    
    # write point features
    if os.path.exists(pt_output_path):
        features = get_feature_from_geojson(pt_output_path)    
        write_pt_into_gpkg(db_instance, features[:], map_name)
    
    # write line features
    if os.path.exists(ln_output_path):
        ln_files = os.listdir(ln_output_path)
        for ln_geojson in ln_files:
            if '.geojson' not in ln_geojson:
                continue
            ln_geojson_path = os.path.join(ln_output_path, ln_geojson)
            features = get_feature_from_geojson(ln_geojson_path)
            write_ln_into_gpkg(db_instance, features[:], map_name)

if __name__ == '__main__':
    write_gpkg('', '', '', '', '','', '')
        