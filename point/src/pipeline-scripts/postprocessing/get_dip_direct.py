import json
import os
from shapely import geometry
import argparse 
from pathlib import Path
from shapely import LineString, MultiPoint, Polygon,Point
import shutil

def get_dip_direction(interm_output_dir_per_map,final_output_per_map):
    interm_output_list = os.listdir(interm_output_dir_per_map)
    map_name = os.path.basename(Path(interm_output_dir_per_map))
    strike_geojson = os.path.join(interm_output_dir_per_map,map_name+'_strike.geojson')
    st_f = open(strike_geojson,'r')
    stk_output = json.load(st_f)
    st_f.close()
    strike_list = ['inclined_bedding','inclined_flow_banding','inclined_metamorphic','overturned_bedding']
    if os.path.isfile(strike_geojson):
        for each_pt in strike_list:
            od_geojson = os.path.join(interm_output_dir_per_map,map_name+'_'+each_pt+'.geojson')
            out_file_path =  os.path.join(final_output_per_map,map_name+'_'+each_pt+'.geojson')
            if os.path.isfile(od_geojson):
                pt_f = open(od_geojson)
                pnt_output = json.load(pt_f)
                for stk_feature in stk_output["point_features"]:
                # dip direction value from each instance
                    dd = stk_feature["features"][0]["properties"]["dip_direction"]
                    cnt_pt_st  = Point(stk_feature["features"][0]["geometry"]["coordinates"])
                    radius = 20
                    st_circle = cnt_pt_st.buffer(radius)
                    min_dist = 10000
                    min_id = 0          
                    print(dd,cnt_pt_st)

                    for pt_feature in pnt_output["point_features"]:
                        cnt_pt = Point(pt_feature["features"][0]["geometry"]["coordinates"])  
                        if st_circle.contains(cnt_pt):
                            dist = cnt_pt_st.distance(cnt_pt)
                            if min_dist > dist:
                                min_dist = dist
                                min_id = pt_feature["features"][0]["id"]
                                

                    for pt_feature in pnt_output["point_features"]:
                        if min_dist != 10000 and (int)(pt_feature["features"][0]["id"]) == (int)(min_id):
                            pt_feature["features"][0]["properties"]["dip_direction"] = dd


                with open(out_file_path, "w") as out_f:
                    json.dump(pnt_output,out_f,indent=1) 

        for each_output in interm_output_list:
            other_pt_name = each_output.replace(map_name+'_','')
            other_pt_name = other_pt_name.replace('.geojson','')         
            if (not each_output.endswith('_strike.geojson') ) and (other_pt_name not in strike_list):
                stitch_out = os.path.join(interm_output_dir_per_map,each_output)
                final_out = os.path.join(final_output_per_map,each_output)
                shutil.copy(stitch_out, final_out)

    else:
        shutil.copytree(interm_output_dir_per_map, final_output_per_map)