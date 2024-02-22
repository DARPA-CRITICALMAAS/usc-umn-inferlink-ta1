#remove point symbol outputs from map lengend. 
import json
import os
from shapely import geometry
import numpy as np
                
       
def retrieve_map_region_bb(metadata_path, map_name):
    map_content_box = None
    for metadata_file in os.listdir(metadata_path):
        if map_name in metadata_file:
            with open(os.path.join(metadata_path, metadata_file), 'r') as f:
                meta = json.load(f)
                if map_content_box is None:  
                    map_content_box = meta.get("map_content_box")

    if map_content_box is not None:
        x, y, w, h = map_content_box
        return geometry.Polygon([[x, y], [x+w, y], [x+w, (y+h)], [x, (y+h)]])
    else:
        return None
        
    
def retrieve_spotting_poly_output(spotting_file, map_region_bb, if_revert_on_y=True):
    # the spotting output is for on visualization (reverse-y)
    if not os.path.exists(spotting_file):
        print("Spotting file NOT found:", spotting_file)
        return []
        
    filtered_spotting_data = []    
    with open(spotting_file) as f:
        spotting_data = json.load(f)
        
    for feature in spotting_data["features"]:
        text = feature["properties"]["text"]
        poly = np.array(feature["geometry"]["coordinates"][0])
        poly[:, 1] = -poly[:, 1]
        poly = geometry.Polygon(poly)
        if not text.isdigit():
            if map_region_bb.contains(poly):
                filtered_spotting_data.append(poly)
    return filtered_spotting_data

    
def check_pt_in_map_region(point, map_region_bb):
    if map_region_bb is None:
        return True    
    return map_region_bb.contains(point)

    
def check_pt_in_text_polys(point, text_polys):
    for text_poly in text_polys:
        if text_poly.contains(point):
            return False
    return True

    
def postprocessing(geojson_path, metadata_path,spotting_path,final_output_dir,
                geojson_visual_flag=False, 
                if_filter_by_map_region=True,
                if_filter_by_text_regions=False):
    #input : per map dir 
    for geojson_file in os.listdir(geojson_path):
        map_name =os.path.basename(geojson_path)
        pnt_name= geojson_file.replace(map_name+"_",'')
        pnt_name=pnt_name.split('.')[0]
        print(map_name,pnt_name)
        
        # retrieve map region bb
        map_region_bb = retrieve_map_region_bb(metadata_path, map_name)
        
        # retrieve text spotting polys
        if spotting_path is not None:
            spotting_file = os.path.join(spotting_path, map_name + '.geojson')
            text_polys = retrieve_spotting_poly_output(spotting_file, map_region_bb, if_revert_on_y=True)
        else:
            text_polys = []
               
        with open(os.path.join(geojson_path, geojson_file), 'r') as f:
            geojson_data = json.load(f)
            
        # create new geojson
        new_geojson_data = geojson_data.copy()
        new_features = []
        for feature in geojson_data["features"]:
            x, y = feature["geometry"]["coordinates"]
            if geojson_visual_flag: y = -y;
            point = geometry.Point(x, y)
            
            KEEP = True
            if if_filter_by_map_region and not check_pt_in_map_region(point, map_region_bb):
                KEEP = False
            
            if if_filter_by_text_regions and not check_pt_in_text_polys(point, text_polys):
                KEEP = False
                
            if KEEP: 
                feature["properties"]["id"]=len(new_features)+1
                new_features.append(feature)       
        new_geojson_data["features"] = new_features    
        final_output_dir_per_map = os.path.join(final_output_dir, map_name)
        if not os.path.isdir(final_output_dir_per_map):
            os.mkdir(final_output_dir_per_map)
        new_geojson_file = os.path.join(final_output_dir_per_map,map_name +"_"+pnt_name+'.geojson')
        with open(new_geojson_file, 'w', encoding='utf8') as f:
            json.dump(new_geojson_data, f, ensure_ascii=False)
    