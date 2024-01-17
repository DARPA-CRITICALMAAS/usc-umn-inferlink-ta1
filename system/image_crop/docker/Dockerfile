# docker pull pytorch/pytorch
# docker build -t mpg4 -f Dockerfile.mpg4 .
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

# INCLUDEX perms.txt

COPY . /ta1/dev/usc-umn-inferlink-ta1/

CMD []
ENTRYPOINT ["python", "/ta1/dev/usc-umn-inferlink-ta1/system/image_crop/map2patch.py"]

