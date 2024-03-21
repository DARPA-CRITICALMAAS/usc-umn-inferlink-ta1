import os
import time
import json
from criticalmaas.ta1_geopackage import GeopackageDatabase
import concurrent.futures
import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiPolygon, Point, MultiLineString, Polygon, LineString
from pytest import mark
from gpkg_helper import reverse_geom_coords

poly_feat_id, poly_type_id, geo_unit_id, ln_type_id, ln_feat_id, pt_feat_id, pt_type_id = 0, 0, 0, 0, 0, 0, 0

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
#     print(db.enum_values('point_type'))
    return db


@mark.parametrize("engine", ["fiona", "pyogrio"])
def write_poly_into_gpkg(db, feat_list, map_name, crs="CRITICALMAAS:pixel"):     
    engine = 'pyogrio' 
    global poly_feat_id, poly_type_id, geo_unit_id
    geo_unit_list = []
    poly_type_list = []
    poly_geom, poly_property = [], []
    
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
        
    geo_unit_feat = {'id': str(geo_unit_id), 'name':feat['properties']['GeologicUnit']['name'],'description':feat['properties']['GeologicUnit']['description'], 'age_text':feat['properties']['GeologicUnit']['age_text'],'t_interval':feat['properties']['GeologicUnit']['t_interval'], 'b_interval':feat['properties']['GeologicUnit']['b_interval'], 't_age': feat['properties']['GeologicUnit']['t_age'] if feat['properties']['GeologicUnit']['t_age'] and feat['properties']['GeologicUnit']['t_age'].isdigit() else None, 'b_age':feat['properties']['GeologicUnit']['b_age'] if feat['properties']['GeologicUnit']['b_age'] and feat['properties']['GeologicUnit']['b_age'].isdigit() else None,'lithology':feat['properties']['GeologicUnit']['lithology']}

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
    gdf = gpd.GeoDataFrame(df_poly_geom) #, crs=crs
    gdf.to_file(
        db.file,
        layer="polygon_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    return
        
def write_pt_into_gpkg(db, feat_list, map_name, crs="CRITICALMAAS:pixel"):   
    global pt_type_id, pt_feat_id
    engine = 'pyogrio'
    pt_type_list = []
    pt_feat_list = []
    
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
                    "type": str(cur_pt_type_id),
                    "dip_direction": 0,
                    "dip": 0,  
                    "confidence": None,
                    "provenance": 'modelled',
                    "geometry": geo_pt
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
    gdf = gpd.GeoDataFrame(df_pt_geom) #, crs=crs
    gdf.to_file(
        db.file,
        layer="point_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    
    return    
        
        
def write_ln_into_gpkg(db, feat_list, map_name, crs="CRITICALMAAS:pixel"): 
    global ln_type_id, ln_feat_id
    engine = 'pyogrio'
    ln_type_list, ln_feat_list = [], []
      
    for ind, feat in enumerate(feat_list):
        if feat['geometry']['type'] == 'MultiLineString':
            ln_geom = reverse_geom_coords(MultiLineString(feat['geometry']['coordinates']))
        else:
            ln_geom = MultiLineString([reverse_geom_coords(LineString(feat['geometry']['coordinates']))])
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
    gdf = gpd.GeoDataFrame(df_ln_geom)
    gdf.to_file(
        db.file,
        layer="line_feature",
        driver="GPKG",
        mode="a",
        engine=engine,
        promote_to_multi=True,
    )
    return

    
def write_gpkg(output_dir, map_name,layout_output_path, poly_output_path, ln_output_path, pt_output_path, logger):    
    
    with open(layout_output_path, 'r') as json_file:
        legend_item_descr_json = json.load(json_file)
        
    out_gpkg_path = f'{output_dir}/{map_name}_img.gpkg'    
    try:
        db_instance = create_gpkg(out_gpkg_path, legend_item_descr_json, map_name)
        logger.info(f'Created a gpkg in the image-coordinate')
    except sqlalchemy.exc.IntegrityError as error:
        logger.error(f'{out_gpkg_path} exists. Please delete it.')
        sys.exit(1)
    
    # write polygon features    
    if os.path.exists(poly_output_path):
        poly_files = os.listdir(poly_output_path)
        for i, poly_geojson in enumerate(poly_files):
            if '.geojson' not in poly_geojson:
                continue
            geojson_path = os.path.join(poly_output_path, poly_geojson)
            features = get_feature_from_geojson(geojson_path)    
            logger.info(f'Writing {i+1}/{len(poly_files)} polygon GeoJson into the img GPKG')
            write_poly_into_gpkg(db_instance, features[:], map_name)
    logger.info(f'All polygon GeoJson is written into the img GPKG')                
    
    # write point features
    if os.path.exists(pt_output_path):
        pt_files = os.listdir(pt_output_path)
        for i, pt_geojson in enumerate(pt_files):
            if '.geojson' not in pt_geojson:
                continue  
            geojson_path = os.path.join(pt_output_path, pt_geojson)
            features = get_feature_from_geojson(geojson_path)    
            logger.info(f'Writing {i+1}/{len(pt_files)} point GeoJson into the img GPKG')
            write_pt_into_gpkg(db_instance, features[:], map_name)
    logger.info(f'All point GeoJson is written into the img GPKG')
    
#     write line features
    if os.path.exists(ln_output_path):
        ln_files = os.listdir(ln_output_path)
        for i, ln_geojson in enumerate(ln_files):
            if '.geojson' not in ln_geojson:
                continue
            ln_geojson_path = os.path.join(ln_output_path, ln_geojson)
            features = get_feature_from_geojson(ln_geojson_path)
            logger.info(f'Writing {i+1}/{len(ln_files)} line GeoJson into the img GPKG')
            write_ln_into_gpkg(db_instance, features[:], map_name)
    logger.info(f'All line GeoJson is written into the img GPKG')
# if __name__ == '__main__':
#     write_gpkg('', '', '', '', '','', '')
        