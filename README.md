# Map Feature Extraction and Georeferencing

## Summary
This repository houses the collaborative efforts of the University of Southern California (USC), the University of Minnesota (UMN), and the InferLink Corporation as part of DARPA's [CRITICALMAAS program](https://knowledge-computing.github.io/criticalmaas-web/) . 
The project aims to develop advanced techniques for automated line, point, and polygon feature extraction for critical mineral symbols on historical maps, as well as the georeferencing of geologic maps.

```
.
│
├── georeferencing/       # Composed of text-based and image-based approaches. Text-based approach is integrated to the system, image-based is under development
│  
├── line/                 # A transformer-based model to detect linear objects from maps
│
├── point/                # An object-detection approach based on YOLO to extract point symbols from maps
│ 
├── polygon/              # Customized DNN model to segment polygon features from maps
│ 
├── segmentation/         # Composed of legend item segmentation and text description extraction
|
└── system/               # Interface code to connect all the above modules
```

## Installation & Usage

Detailed installation and usage guide can be found in each module's read-me file.
* [Georeferencing](https://github.com/DARPA-CRITICALMAAS/usc-umn-inferlink-ta1/tree/main/georeferencing#readme)
* [Line](https://github.com/DARPA-CRITICALMAAS/usc-umn-inferlink-ta1/tree/main/line#readme)
* [Point](https://github.com/DARPA-CRITICALMAAS/usc-umn-inferlink-ta1/blob/main/point/src/pipeline-scripts/README.md)
* [Polygon](https://github.com/DARPA-CRITICALMAAS/usc-umn-inferlink-ta1/blob/main/polygon/README.md)
* [Legend Item Segmentation](https://github.com/DARPA-CRITICALMAAS/usc-umn-inferlink-ta1/blob/main/segmentation/legend_item_segmentation/README.md) and [Legend Item Description Extraction](https://github.com/DARPA-CRITICALMAAS/usc-umn-inferlink-ta1/blob/main/segmentation/legend_item_description_segment/README.md)
* [System](https://github.com/DARPA-CRITICALMAAS/usc-umn-inferlink-ta1/blob/main/system/README.md)


## License
The contents of this repository are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en). Users are encouraged to review the licenses associated with specific components before usage or redistribution.
