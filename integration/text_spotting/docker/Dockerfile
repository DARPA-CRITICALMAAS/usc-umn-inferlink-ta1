FROM knowledgecomputing/mapkurator_recogito_2023:latest

COPY . /ta1/dev/usc-umn-inferlink-ta1/

RUN cp /ta1/dev/usc-umn-inferlink-ta1/system/mapkurator/mapkurator-system/run_text_spotting.py /home/mapkurator-system/

# INCLUDEX perms.txt

CMD []
ENTRYPOINT [  "/ta1/dev/usc-umn-inferlink-ta1/integration/text_spotting/docker/runme.sh" ]

