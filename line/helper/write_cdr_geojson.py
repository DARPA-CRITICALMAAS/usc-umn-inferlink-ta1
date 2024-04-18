import os
import json
from shapely.geometry import MultiLineString, LineString

def geometry_to_coordinates(geometry):
    if isinstance(geometry, MultiLineString):
        return [list(line.coords) for line in geometry]
    elif isinstance(geometry, LineString):
        line_coords = [list(coord) for coord in list(geometry.coords)]
        return line_coords
    else:
        raise ValueError("Unsupported geometry type")

def write_geojson_cdr(output_geojson_path, all_lines, legend_text=None, feature_name=None):
    line_features = []
    cnt = 0   
    for line_cat, lines in all_lines.items():
        for line_geom in lines:
            if isinstance(line_geom, MultiLineString):
                for _line_geom in line_geom:
                    cnt += 1
                    line_coords = geometry_to_coordinates(_line_geom)
                    line_feat = {
                        "id": cnt,
                        "geometry": {
                            "coordinates": line_coords
                        }, 
                        "properties":{
                            "model": "umn-usc-inferlink",
                            "model_version": "0.0.1",
                            "confidence": None,
                            "dash_pattern": line_cat,
                            "symbol": None
                        }
                    }
                    line_features.append(line_feat)
            else:
                cnt += 1
                line_coords = geometry_to_coordinates(line_geom)
                line_feat = {
                    "id": cnt,
                    "geometry": {
                        "coordinates": line_coords
                    }, 
                    "properties":{
                        "model": "umn-usc-inferlink",
                        "model_version": "0.0.1",
                        "confidence": None,
                        "dash_pattern": line_cat,
                        "symbol": None
                    }
                }
                line_features.append(line_feat)
                
    cdr_line_data = {
        "id": "0",
        "crs": "CRITICALMAAS:pixel",
        "cdr_projection_id": "",
        "name": feature_name,
        "description": legend_text,
        "legend_bbox": [],
         "line_features": {
             "features": line_features
         }
    }
    with open(output_geojson_path, "w") as f:
        json.dump(cdr_line_data, f)
    return 
    