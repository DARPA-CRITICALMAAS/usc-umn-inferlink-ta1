The integration code for mapKurator and recogito sits in three files - 
/home/mapkurator-system/recogito_integration/process_image.py
/home/mapkurator-system/run_pipeline.py
/home/recogito2/app/transform/mapkurator/MapKuratorService.scala

Given below are the arguments for running process_image.py as a unit.
The format of arguments shown in this document is to pass arguments in the launch.json when python debugger is enabled in VSCode. 
A sample for launch.json is - 
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "subProcess": true,
            "args": [  
                "iiif",
                "--url=https://map-view.nls.uk/iiif/2/12563%2F125635459/info.json",
                "--dst=data/test_imgs/sample_output/",
                "--filename=sample_file_iiif",
                "--text_spotting_model_dir=/home/spotter_v2/PALEJUN/",
                "--spotter_model=spotter_v2",
                "--spotter_config=/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml",
                "--gpu_id=3"
            ],
            "justMyCode": true
        }
    ]
}

To use the commandline edit the samples to match commandline calls as - 
python process_image.py iiif --url https://map-view.nls.uk/iiif/2/12563%2F125635459/info.json   
--dst data/test_imgs/sample_output/   
--filename sample_file_iiif   
--text_spotting_model_dir /home/spotter_v2/PALEJUN/    
--spotter_model spotter_v2    
--spotter_config /home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml    
--gpu_id 3   

Sample input commands with spotter_v2 : 

Sample test input for iiif -
"iiif",
"--url=https://map-view.nls.uk/iiif/2/12563%2F125635459/info.json",
"--dst=data/test_imgs/sample_output/",
"--filename=sample_file_iiif",
"--text_spotting_model_dir=/home/spotter_v2/PALEJUN/",
"--spotter_model=spotter_v2",
"--spotter_config=/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml",
"--gpu_id=3"

Sample test input for image upload -      
"file",
"--src=/home/mapkurator-test-images/test.jpeg",
"--dst=data/test_imgs/sample_output/",
"--filename=sample_file_upload",
"--text_spotting_model_dir=/home/spotter_v2/PALEJUN/",
"--spotter_model=spotter_v2",
"--spotter_config=/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml",
"--gpu_id=3"

Sample test input for wmts upload -
"wmts",
"--url=https://wmts.maptiler.com/aHR0cDovL3dtdHMubWFwdGlsZXIuY29tL2FIUjBjSE02THk5dFlYQnpaWEpwWlhNdGRHbHNaWE5sZEhNdWN6TXVZVzFoZW05dVlYZHpMbU52YlM4eU5WOXBibU5vTDNsdmNtdHphR2x5WlM5dFpYUmhaR0YwWVM1cWMyOXUvanNvbg/wmts",
"--boundary={\"type\":\"Feature\",\"properties\":{},\"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[-1.1248,53.9711],[-1.0592,53.9711],[-1.0592,53.9569],[-1.1248,53.9569],[-1.1248,53.9711]]]}}",
"--zoom=16",
"--dst=data/test_imgs/sample_output/",
"--filename=sample_file_wmts",
"--coord=epsg4326",
"--text_spotting_model_dir=/home/spotter_v2/PALEJUN/",
"--spotter_model=spotter_v2",
"--spotter_config=/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml",
"--gpu_id=3"

Sample input commands with spotter_testr: 

Sample test input for iiif -
"iiif",
"--url=https://map-view.nls.uk/iiif/2/12563%2F125635459/info.json",
"--dst=data/test_imgs/sample_output/",
"--filename=sample_file_iiif",
"--text_spotting_model_dir=/home/spotter_testr/TESTR/",
"--spotter_model=testr",
"--spotter_config=/home/spotter_testr/TESTR/configs/TESTR/SynthMap/SynthMap_Polygon.yaml",
"--gpu_id=3"

Sample test input for image upload -      
"file",
"--src=/home/mapkurator-test-images/test.jpeg",
"--dst=data/test_imgs/sample_output/",
"--filename=sample_file_upload",
"--text_spotting_model_dir=/home/spotter_testr/TESTR/",
"--spotter_model=testr",
"--spotter_config=/home/spotter_testr/TESTR/configs/TESTR/SynthMap/SynthMap_Polygon.yaml",
"--gpu_id=3"

Sample test input for wmts upload -
"wmts",
"--url=https://wmts.maptiler.com/aHR0cDovL3dtdHMubWFwdGlsZXIuY29tL2FIUjBjSE02THk5dFlYQnpaWEpwWlhNdGRHbHNaWE5sZEhNdWN6TXVZVzFoZW05dVlYZHpMbU52YlM4eU5WOXBibU5vTDNsdmNtdHphR2x5WlM5dFpYUmhaR0YwWVM1cWMyOXUvanNvbg/wmts",
"--boundary={\"type\":\"Feature\",\"properties\":{},\"geometry\":{\"type\":\"Polygon\",\"coordinates\":[[[-1.1248,53.9711],[-1.0592,53.9711],[-1.0592,53.9569],[-1.1248,53.9569],[-1.1248,53.9711]]]}}",
"--zoom=16",
"--dst=data/test_imgs/sample_output/",
"--filename=sample_file_wmts",
"--coord=epsg4326",
"--text_spotting_model_dir=/home/spotter_testr/TESTR/",
"--spotter_model=testr",
"--spotter_config=/home/spotter_testr/TESTR/configs/TESTR/SynthMap/SynthMap_Polygon.yaml",
"--gpu_id=3"

Recogito Commands for Spotter V2 : 
IIIF
val cli = s"python /home/mapkurator-system/recogito_integration/process_image.py iiif --url=$filename --dst=data/test_imgs/sample_output/ --filename=${part.getId} --text_spotting_model_dir=/home/spotter_v2/PALEJUN/ --spotter_model=spotter_v2 --spotter_config=/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml --gpu_id=3" 

FILE 
var cli = s"python /home/mapkurator-system/recogito_integration/process_image.py file --src=$dockerPath --dst=/home/mapkurator-system/data/test_imgs/sample_output/ --filename=${part.getId}  --text_spotting_model_dir=/home/spotter_v2/PALEJUN/ --spotter_model=spotter_v2 --spotter_config=/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml --gpu_id=3"

WMTS
var cli = s"python /home/mapkurator-system/recogito_integration/process_image.py wmts --url=$filename --boundary=$bounds --zoom=16 --dst=data/test_imgs/sample_output/ --filename=${part.getId} --coord=epsg4326  --text_spotting_model_dir=/home/spotter_v2/PALEJUN/ --spotter_model=spotter_v2 --spotter_config=/home/spotter_v2/PALEJUN/configs/PALEJUN/SynthMap/SynthMap_Polygon.yaml --gpu_id=3"

Recogito Commands for Spotter_testr: 
IIIF
val cli = s"python /home/mapkurator-system/recogito_integration/process_image.py iiif --url=$filename --dst=data/test_imgs/sample_output/ --filename=${part.getId} --text_spotting_model_dir=/home/spotter_testr/TESTR/ --spotter_model=testr --spotter_config=/home/spotter_testr/TESTR/configs/TESTR/SynthMap/SynthMap_Polygon.yaml --gpu_id=3" 

FILE 
var cli = s"python /home/mapkurator-system/recogito_integration/process_image.py file --src=$dockerPath --dst=/home/mapkurator-system/data/test_imgs/sample_output/ --filename=${part.getId}  --text_spotting_model_dir=/home/spotter_testr/TESTR/ --spotter_model=testr --spotter_config=/home/spotter_testr/TESTR/configs/TESTR/SynthMap/SynthMap_Polygon.yaml --gpu_id=3"

WMTS
var cli = s"python /home/mapkurator-system/recogito_integration/process_image.py wmts --url=$filename --boundary=$bounds --zoom=16 --dst=data/test_imgs/sample_output/ --filename=${part.getId} --coord=epsg4326  --text_spotting_model_dir=/home/spotter_testr/TESTR/ --spotter_model=testr --spotter_config=/home/spotter_testr/TESTR/configs/TESTR/SynthMap/SynthMap_Polygon.yaml --gpu_id=3"