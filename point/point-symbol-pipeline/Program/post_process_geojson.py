#remove point symbol outputs from map lengend. 
import json
import os
from shapely import geometry
import argparse 

#name convention
# metadata - {map_name}_point.json
# geojson -  map_name.geojson

def remove_pnts_from_legend(metadata_path,geojson_path,visual_flag=False):
    for each_json in os.listdir(metadata_path):
        f=open(os.path.join(metadata_path,each_json))
        dict_per_map=json.load(f)
        map_region_bb=dict_per_map["map_content_box"]
        # print(each_json)
        if map_region_bb != None:
            for val_img in os.listdir(geojson_path):
                map_name=val_img.split('.')[0]
                if map_name in each_json:
                    # print(map_name,each_json)
                    file_path = os.path.join(geojson_path, val_img)
                    f = open(file_path)
                    data = json.load(f)
                    new_data_feature=[]
                    for feature in data["features"]:
                        point = geometry.Point(feature["geometry"]["coordinates"][0], feature["geometry"]["coordinates"][1])
                        x, y, w, h = map_region_bb
                        if visual_flag:                       
                            bb_list=[[x, -y], [x+w, -y], [x+w, -(y+h)], [x, -(y+h)]]
                        else:
                            bb_list=[[x, y], [x+w, y], [x+w, (y+h)], [x, (y+h)]]
                        polygon = geometry.Polygon(bb_list)
                        if polygon.contains(point) :
                            new_data_feature.append(feature)          
                    data["features"]=new_data_feature           
                    new_filepath = os.path.join(geojson_path, val_img)
                    with open(new_filepath, 'w', encoding='utf8') as f:
                        json.dump(data, f, ensure_ascii=False)
