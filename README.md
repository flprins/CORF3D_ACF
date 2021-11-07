# CORF3D contour maps with application to Holstein cattle recognition using RGB and thermal images

This respository contains code for the project "CORF3D contour maps with application to Holstein cattle recognition using RGB and thermal images"

#### Summary

* [Introduction](#Introduction)
* [Getting Started](#Getting-started)
* [Dataset](#Dataset)
* [Repository Structure](#Repository-Structure)
* [Running Experiments](#Running-Experiments)
* [Performance](#Performance)
* [Acknowledgment](#Acknowledgment)
* [Citation](#Citation) 

## Introduction

Livestock management involves the monitoring of farm animals by tracking certain physiological and phenotypical characteristics over time. In the dairy industry, for instance, cattle are typically equipped with RFID ear tags. The corresponding data (e.g. milk properties) can then be automatically assigned to the respective cow when they enter the milking station. In order to move towards a more scalable, affordable, and welfare-friendly approach, automatic non-invasive solutions are more desirable. Thus, a non-invasive approach is proposed in this paper for the automatic identification of individual Holstein cattle from the side view while exiting a milking station. It considers input images from a thermal-RGB camera. The thermal images are used to delineate the cow from the background. Subsequently, any occluding rods from the milking station are removed and inpainted with the fast marching algorithm. Then, it extracts the RGB map of the segmented cattle along with a novel CORF3D contour map. The latter contains three contour maps extracted by the Combination of Receptive Fields (CORF) model with different strengths of push-pull inhibition. This mechanism suppresses noise in the form of grain type texture. The effectiveness of the proposed approach is demonstrated by means of experiments using a 5-fold and a leave-one day-out cross-validation on a new data set of 3694 images of 383 cows collected from the Dairy Campus in Leeuwarden (the Netherlands) over 9 days. In particular, when combining RGB and CORF3D maps by late fusion, an average accuracy of 99.64% (±0.13) was obtained for the 5-fold cross validation and 99.71% ( ± 0.31) for the leave--one day--out experiment. The two maps were combined by first learning two ConvNet classification models, one for each type of map. The feature vectors in the two FC layers obtained from training images were then concatenated and used to learn a
linear SVM classification model. In principle, the proposed approach with the novel CORF3D contour maps is suitable for various image classication applications, especially where texture is a nuisance.

## Getting started

In order to run this repository, we advise you to install python 3.6 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install requirments on it:

```bash
conda create --name CORF3D_HCR python=3.6
```

Clone this repository:

```bash
git clone https://github.com/ameybhole/CAIIHC.git 
```

Install requirements:

```bash
$ pip install -r requirements.txt
```

## Dataset

The pre-processing steps used were based on the following paper:

[A. Bhole, O. Falzon, M. Biehl, G. Azzopardi, “A Computer Vision Pipeline that Uses Thermal and RGB Images for the Recognition of Holstein Cattle”, Computer Analysis of Images and Patterns (CAIP), pp. 108-119, 2019](https://link.springer.com/chapter/10.1007/978-3-030-29891-3_10)

Github link to the code for the above paper: 

- Link: https://github.com/ameybhole/IIHC

## Repository Structure

```
.
├── src                        # source code of the project 
|   ├── CORFpushpull           # source code for CORFpushpull model
|   ├── Preprocessing          # scripts for segmentation
|   ├── Temp_extraction        # source code for temperature extraction 
|   ├── data_load.py           # script for data loading
|   ├── data_preprocessing.py  # script for batch data preprocesing
|   ├── evaluation.py          # script for model evaluation 
|   ├── feature_maps.py        # script for feature map generation   
|   ├── models.py              # script for compiling model
|   ├── train.py               # script to train model
|   └──visualizations.py       # script for visualization
└── main.py                    # script to run exepriments with different parameters settings
```

## Running Experiments

### Train 

```Bash

```

## Performance

#### Average performance of three ConvNets and four feature sets across 5-fold cross-validation

| __Model__ | __Feature set__ | __Test accuracy__ (\%) | __F1__ (\%) |
|:--------------:|:--------------------:|:---------------------------:|:----------------:|
| MobileNet      | RGB                  | 98.18 ± 0.28            | 97.75 ± 0.25 |
|                | MSX                  | 97.04 ± 0.37            | 96.88 ± 0.42 |
|                | Temp3D               | 50.71 ± 1.34            | 50.15 ± 1.24 |
|                | CORF3D               | 99.16 ± 0.21            | 99.08 ± 0.18 |
| DenseNet121    | RGB                  | 98.56 ± 0.28            | 98.51 ± 0.30 |
|                | MSX                  | 95.38 ± 0.69            | 95.34 ± 0.55 |
|                | Temp3D               | 13.12 ± 4.23            | 6.95 ± 4.56   |
|                | CORF3D               | 97.88 ± 0.27            | 98.27 ± 0.25  |
| Xception       | RGB                  | 98.36 ± 0.28            | 98.34 ± 0.24  |
|                | MSX                  | 98.36 ± 0.28            | 98.37 ± 0.36 |
|                | Temp3D               | 9.38 ± 2.34             | 6.01 ± 3.68  |
|                | CORF3D               | 98.86 ± 0.31            | 98.75 ± 0.35  |


#### Average performance of different fusions of feature sets across 5-fold cross-validation

| __Model__             | __Feature set__           | __Test accuracy__ (\%) | __F1__ (\%)}          |
|:--------------------------:|:------------------------------:|:---------------------------:|:-------------------------:|
| MobileNet + SVM            | RGB + MSX                      | 98.68 ± 0.35            | 98.48 ± 0.38          |
|                            | RGB + CORF3D                   | 99.19 ± 0.11            | 99.11 ± 0.15          |
| __DenseNet121 + SVM__      | RGB + MSX                      | 99.00 ± 0.41            | 98.96 ± 0.48          |
|                            | __RGB + CORF3D__               | __99.64 ± 0.13__        | __99.54 ± 0.16__      |
| Xception + SVM             | RGB + MSX                      | 98.1 ±  0.14            | 98.08 ± 0.18          |
|                            | RGB + CORF3D                   | 99.29 ± 0.37            | 99.24 ± 0.33          |

#### Average performance of different feature sets and models across a leave-one-day-out cross-validation

| __Model__             | __Feature set__  | __Test accuracy__ (\%) | __F1__ (\%)          |
|:--------------------------:|:---------------------:|:---------------------------:|:-------------------------:|
| MobileNet                  | RGB                   | 98.01 ± 0.39            | 98.08 ± 0.45          |
| MobileNet                  | CORF3D                | 98.43 ± 0.47            | 99.55 ± 0.26          |
| MobileNet + SVM            | RGB + CORF3D          | 99.64 ± 0.40            | 99.44 ± 0.42          |
| DenseNet121                | RGB                   | 98.86 ± 0.45            | 98.75 ± 0.43          |
| DenseNet121                | CORF3D                | 98.85 ± 0.39            | 98.76 ± 0.50          |
| __DenseNet121 + SVM__      | __RGB + CORF3D__      | __99.71 ± 0.30__        | __99.68 ± 0.30__      |
| Xception                   | RGB                   | 98.46 ± 0.64            | 98.66 ± 0.28          |
| Xception                   | CORF3D                | 98.60 ± 0.53            | 98.59 ± 0.26          |
| Xception + SVM             | RGB + CORF3D          | 99.15 ± 0.35            | 99.21 ± 0.33          |

## Acknowledgment

We thank the Dairy Campus in Leeuwarden for permitting the data collection used in this project and for approving its availability for academic use. We are also grateful for the discussions held with Prof. Michael Biehl from the University of Groningen who contributed in the designing stage of several aspects of the project. Finally, we thank the Center for Information Technology of the University of Groningen for their support and for providing access to the Peregrine high performance computing cluster.

## Citation

If you find this paper or code useful, we encourage you to cite the paper. BibTeX:

