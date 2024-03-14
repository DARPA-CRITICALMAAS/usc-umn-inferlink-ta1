from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class Geometry(BaseModel):
    type: str

class Polygon(Geometry):
    """
    coordinates in polygon are (col, row).
    """
    coordinates: List[List[List[float]]]
    type: str = Field(default="Polygon")

class PolygonType(BaseModel):
    id: Optional[int]=None
    name:  Optional[str]=None
    color: Optional[str] = Field(default=None, description= "color is Hex_color_code") 
    pattern: Optional[str]=None
    abbreviation: Optional[str]=None
    description: Optional[str]=None
    category: Optional[str]=None    

class GeologicUnit(BaseModel):
    name: Optional[str]=None
    description: Optional[str]=None
    comments: Optional[str]=None
    age_text: Optional[str]=None
    t_interval: Optional[str]=None
    b_interval: Optional[str]=None
    t_age: Optional[int]=None
    b_age: Optional[int]=None
    lithology: Optional[List[str]]=None

class Property(BaseModel):
    id: int
    name: str = Field(default="PolygonFeature", Literal=True)
    PolygonType: PolygonType
    GeologicUnit: GeologicUnit
    

class Feature(BaseModel):
    type: str = "Feature"
    geometry: Polygon
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
    
    geojson_path = '/data/weiweidu/criticalmaas_data/hackathon2/nickel_polygon_extraction_demo/24651_2360/24651_2360_0_poly_PolygonFeature.geojson'
    
    # Load the GeoJSON file
    geojson_file_path = Path(geojson_path)
    geojson_data = json.loads(geojson_file_path.read_text())

    # Parse the GeoJSON data into the Pydantic model
    feature_collection = FeatureCollection.parse_obj(geojson_data)

    # Now, you can access the parsed data
    print(feature_collection.features[0].properties.GeologicUnit) 