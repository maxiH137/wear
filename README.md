# TA-DA! - Improving Activity Recognition using Temporal Adapters and Data Augmentation

<!--<img loop src="teaser.gif" width="100%"/>-->

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3675094.3678454-b31b1b.svg
)](https://dl.acm.org/doi/10.1145/3675094.3678454)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]
## About this repository
This repository was imported from: https://github.com/mariusbock/wear. <br/>
We extended the existing repository with the following approaches:
1. Integration of a Temporal-Informative adapter
2. Data Augmentation 

The files for 1. consist of several extensions in the model files in the camera_baseline folder. <br/>
The file "4_create_features_and_annotations_aug.py" for 2. is placed the data_creation folder and can be used to create augmented data/subjects. 
## Abstract
In this report, we describe the technical details of our submission to
the WEAR Dataset Challenge 2024. For this competition, we use two
approaches to boost the performance of the official WEAR GitHub
repository. 1) Integration of a Temporal-Informative adapter (TIA)
into the models of the WEAR repository; 2) Data Augmentation
Techniques to enrich the provided test dataset. Our method achieves
roughly 4.7% improved results on the test set of the WEAR Dataset
Challenge 2024 compared to the baseline of the WEAR repository.

## Changelog
- 26/09/2024: set the repository to public
## Installation
Please follow instructions mentioned in the [INSTALL.md](/INSTALL.md) file.

## Download
The full dataset can be downloaded [here](https://bit.ly/wear_dataset)

The download folder is divided into 3 subdirectories
- **annotations (> 1MB)**: JSON-files containing annotations per-subject using the THUMOS14-style
- **processed (15GB)**: precomputed I3D, inertial and combined per-subject features
- **raw (130GB)**: Raw, per-subject video and inertial data

## Reproduce Experiments from TA-DA
Once having installed requirements, one can rerun experiments by running the `create_features_and_annotations_mod.py` for data augmentation and the `main.py` script for the experiment:

````
python create_features_and_annotations_mod.py --config ./configs/60_frames_30_stride/tridet_inertial_aug.yaml
````

````
python main.py --config ./configs/60_frames_30_stride/tridet_inertial_aug.yaml --seed 1 --eval_type split
````

Each config file represents one type of experiment. Each experiment was run three times using three different random seeds (i.e. `1, 2, 3`). To rerun the experiments without changing anything about the config files, please place the complete dataset download into a folder called `data/wear` in the main directory of the repository.

## Postprocessing
Please follow instructions mentioned in the [README.md](/postprocessing/README.md) file in the postprocessing subfolder.

### Logging using Neptune.ai

In order to log experiments to [Neptune.ai](https://neptune.ai) please provide `project`and `api_token` information in your local deployment (see lines `34-35` in `main.py`)

## Record your own Data
Please follow instructions mentioned in the [README.md](/data_creation/README.md) file in the data creation subfolder.

### License
TA-DA! is offered under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. You are free to use, copy, and redistribute the material for non-commercial purposes provided you give appropriate credit, provide a link to the license, and indicate if changes were made. If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original. You may not use the material for commercial purposes.

## Contact
 
Helge Hartleb (helge.hartleb@student.uni-siegen.de)

Maximilian Hopp (maximilian2.hopp@student.uni-siegen.de)

## Cite as
```
@inproceedings{10.1145/3675094.3678454,
author = {Hopp, Maximilian and Hartleb, Helge and Burchard, Robin},
title = {TA-DA! - Improving Activity Recognition using Temporal Adapters and Data Augmentation},
year = {2024},
isbn = {9798400710582},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3675094.3678454},
doi = {10.1145/3675094.3678454},
booktitle = {Companion of the 2024 on ACM International Joint Conference on Pervasive and Ubiquitous Computing},
pages = {551–554},
numpages = {4},
keywords = {data augmentation, deep learning, human activity recognition, machine learning, temporal action detection},
location = {Melbourne VIC, Australia},
series = {UbiComp '24}
}
```

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
