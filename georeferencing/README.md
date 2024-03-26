# Text-based Georferencing

## Environment and Dependencies:
The text-based method can run on both CPU-only and GPU machines. The code was tested on a Ubuntu 20.04.1 machine with AMD Ryzen Threadripper PRO 5975WX 32-Cores processor, 512GB (64GB x 8) Memory at 3200 MHz, with 2TB SSD. 

* Data dependency:
  - Create a folder named `support_data` at the same level as `run_georeference.py` 
  - Download the following files to the above folder:
   [Support data download link](https://drive.google.com/drive/folders/17eHH1y71tB_WGizi88d2FLB5U8EujniV)
* Package dependency: 
  ```
    cd text-based
    python3 -m pip install -r requirements.txt
  ```
* Python version: Python3.10 and above


## How to run:

**Method 1**

To run the bare-bone code and process one single image, use `run_georeference.py` and specify the correct `input_path` and `output_path`. Example:
```
python3 run_georenference.py  --input_path='../input_data/CO_Frisco.png' --output_path='../output_georef/CO_Frisco.json'
```

**Method 2**

To run georeferencing web application, execute `streamlit run app.py`. 
