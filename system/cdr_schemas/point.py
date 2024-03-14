from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Geometry(BaseModel):
    type: str

class Point(Geometry):
    """
    coordinates in line are (col, row).
    """
    coordinates: List[int]
    type: str = Field(default="Point")
        

class Property(BaseModel):
    id: int
    type: str = Field(description='The point feature name, such as prospect')
    score: float = Field(description='The predictio probability from the ML model')
    bbox: List[int] = Field(description='The extacted bounding box from the ML model')
#     dip: int
#     dip_direction: int
#     provenance: str = Field(default="modelled")
    
        
class Feature(BaseModel):
    type: str = "Feature"
    geometry: Point
    properties: Property 

class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"   
    features: List[Feature]

if __name__ == '__main__':
    import json
    from pathlib import Path
    
    geojson_path = '/data/weiweidu/criticalmaas_data/hackathon2/nickel_point_extraction_demo/10705_61989/10705_61989_inclined_metamorphic.geojson'
    
    # Load the GeoJSON file
    geojson_file_path = Path(geojson_path)
    geojson_data = json.loads(geojson_file_path.read_text())

    # Parse the GeoJSON data into the Pydantic model
    feature_collection = FeatureCollection.parse_obj(geojson_data)

    # Now, you can access the parsed data
    print(feature_collection.features[0].geometry) 