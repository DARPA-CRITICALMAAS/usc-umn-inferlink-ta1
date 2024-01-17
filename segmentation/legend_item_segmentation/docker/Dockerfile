#FROM nvidia/cuda:11.4.0-base-ubuntu20.04
#FROM pytorch/pytorch

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt update 
RUN \
  DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
  && apt-get -y install apt-utils \
  && apt-get -y install  software-properties-common \
  && conda update -y -n base -c defaults conda

RUN conda install -y -n base \
    scipy \
    matplotlib \
    numpy \
    pandas==2.0.3 \
    geopandas==0.13.2 \
    shapely==2.0.1 \
    conda-forge::gdal==3.6.2 \
    fastai::opencv-python-headless

RUN apt install -y libtiff-dev \
    && apt install -y tesseract-ocr libtesseract-dev \
    && pip --no-input install pytesseract \
    && conda install -y -n base rasterio pyproj

RUN pip install openai

# INCLUDEX perms.txt

COPY . /ta1/dev/usc-umn-inferlink-ta1/

ENV TESSERACT_EXE=/usr/bin/tesseract
CMD []
ENTRYPOINT ["python", "/ta1/dev/usc-umn-inferlink-ta1/segmentation/legend_item_segmentation/map_legend_segmentation.py"]

