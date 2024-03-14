from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Geometry(BaseModel):
    type: str

class Line(Geometry):
    """
    coordinates in line are (row, col).
    """
    coordinates: List[List[float]]
    type: str = Field(default="LineString")

class Property(BaseModel):
    ID: int
    geometr: str = Field(default="line")
    name: str = Field(description="the line feature's name, such as fault line")
    direction: int = Field(default=0, decription='direction range [0, 360]')
    descript: str
    dash: Optional[str] = Field(default=None, description= "values = {solid, dash, dotted}")
    symbol: Optional[str]=None
    
        
class Feature(BaseModel):
    type: str = "Feature"
    geometry: Line
    properties: Property 

class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    crs: Dict[str, Any] = Field(
        description="""
            An example: "crs": { "type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}
        """
    )    
    features: List[Feature]
        
if __name__ == '__main__':
    import json
    from pathlib import Path
    
    geojson_path = '/data/weiweidu/criticalmaas_data/hackathon2/nickel_line_extraction_demo/10705_61989/10705_61989_fault_line_pred.geojson'
    
    # Load the GeoJSON file
    geojson_file_path = Path(geojson_path)
    geojson_data = json.loads(geojson_file_path.read_text())

    # Parse the GeoJSON data into the Pydantic model
    feature_collection = FeatureCollection.parse_obj(geojson_data)

    # Now, you can access the parsed data
    print(feature_collection.features[0].geometry.coordinates) 