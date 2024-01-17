FROM nvidia/cuda:12.3.1-base-ubuntu22.04
#CMD nvidia-smi

FROM pytorch/pytorch

RUN \
  DEBIAN_FRONTEND=noninteractive \
  TZ=Etc/UTC \
  apt update \
  && apt-get -y install tzdata apt-utils software-properties-common \
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

RUN apt install -y libtiff-dev

RUN pip --no-input install scikit-image Pillow==9.4.0 rasterio wandb

RUN apt update \
    && apt -y install libvips-dev \
    && pip --no-input install pyvips

# INCLUDEX perms.txt

COPY . /ta1/dev/usc-umn-inferlink-ta1/

CMD []
ENTRYPOINT ["python", "/ta1/dev/usc-umn-inferlink-ta1/polygon/loam_handler.py"]
