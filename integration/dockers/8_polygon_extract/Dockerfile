FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# INCLUDEX base.txt

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
#ENTRYPOINT ["python", "/ta1/dev/usc-umn-inferlink-ta1/polygon/loam_handler.py"]
ENTRYPOINT ["python", "/ta1/dev/usc-umn-inferlink-ta1/integration/docker-tools/always_pass.py"]
