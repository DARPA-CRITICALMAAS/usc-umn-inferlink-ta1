FROM knowledgecomputing/mapkurator_recogito_2023:latest

COPY . /ta1/dev/usc-umn-inferlink-ta1/

RUN cp \
    /ta1/dev/usc-umn-inferlink-ta1/system/mapkurator/mapkurator-system/run_text_spotting.py \
    /home/mapkurator-system/

RUN cp \
    /ta1/dev/usc-umn-inferlink-ta1/system/mapkurator/mapkurator-system/m3_image_geojson/stitch_output.py \
    /home/mapkurator-system/m3_image_geojson/stitch_output.py

# INCLUDEX perms.txt

CMD []
ENTRYPOINT [  "/ta1/dev/usc-umn-inferlink-ta1/integration/dockers/6_text_spotting/runme.sh" ]
