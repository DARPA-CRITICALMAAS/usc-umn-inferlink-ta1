
## Dependencies:
* Data dependency: create a folder named `support_data` the same level as `run_georeference.py` and download the following files to the folder:
   [Support data download link](https://drive.google.com/drive/folders/17eHH1y71tB_WGizi88d2FLB5U8EujniV)
* Package dependency: `python3 -m pip install -r requirements.txt`
* Python version: Python3.10 and above


## How to run:
To run georeferencing web application, execute `streamlit run app.py`. 

To run the bare-bone code and process one single image, use `run_georeference.py` and specify the correct `input_path` and `output_path`. Example:
```
python3 run_georenference.py  --input_path='../input_data/CO_Frisco.png' --output_path='../output_georef/CO_Frisco.json'
```
