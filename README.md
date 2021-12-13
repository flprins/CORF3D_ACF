# CORF3D contour maps with application to Holstein cattle recognition using RGB and thermal images

This respository contains code for the project "CORF3D contour maps with application to Holstein cattle recognition using RGB and thermal images"

If you have any questions on this repository or the related paper, feel free to [create an issue](https://github.com/ameybhole/CORF3D_HCR/issues/new) or [send me an email](amey.bhole77@gmail.com).
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
https://github.com/ameybhole/CORF3D_HCR.git 
```

Install requirements:

```bash
$ pip install -r requirements.txt
```

Install MATLAB Engine API for Python:

```bash
$ cd "matlabroot\extern\engines\python"
$ python setup.py install
```
For more information regarding MATLAB Engine installation please visit [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)

## Dataset

You can download the datasets [here](https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/7M108F)

We advise you to structure the data in the following way:

```
.
├── data                       # datasets used for the project 
|   ├── Raw                    # Folder with raw datasets
|   |   ├── RGB                # Folder with raw RGB images
|   |   ├── Thermal            # Folder with raw Thermal images 
|   ├── Preprocessed           # Folder with Preprocessed datasets
|   |   ├── RGB                # Folder with preprocessed RGB images
|   |   ├── CORF3D             # Folder with preprocessed CORF3D .npy files 
|   |   ├── TEMP3D             # Folder with preprocessed TEMP3D .npy files   
|   |   ├── MSX                # Folder with preprocessed MSX images
|   └──timestamp.xlsx          # timestamp file
```

## Repository Structure

```
.
├── data                       # Folder with all the datasets
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

#### Train without preprocessing

##### Leave one day out cross-validation

```Bash
python main.py --preprocessing False --method leave_one_day_out --mode fusion --feature_map_1 RGB --feature_map_2 CORF3D --dataset_1 [path to dataset 1] --dataset_2 [path to dataset 2] --num_epochs 15 --resize 224 --batch_size 32 --classes 383 --trainable True --include_top False --model densenet121 
```
##### 5-fold cross-validation

```Bash
python main.py --preprocessing False --method 5_fold --mode fusion --feature_map_1 RGB --feature_map_2 CORF3D --dataset_1 [path to dataset 1] --dataset_2 [path to dataset 2] --num_epochs 15 --resize 224 --batch_size 32 --classes 383 --trainable True --include_top False --learning_rate 0.0005 --model densenet121 
```

#### Train with preprocessing

##### Leave one day out cross-validation

```Bash
python main.py --preprocessing True --method leave_one_day_out --mode fusion --feature_map_1 RGB --feature_map_2 CORF3D --dataset_1 [path to dataset 1] --dataset_2 [path to dataset 2] --num_epochs 15 --resize 224 --batch_size 32 --classes 383 --trainable True --include_top False --model densenet121 
```
##### 5-fold cross-validation

```Bash
python main.py --preprocessing True --method 5_fold --mode fusion --feature_map_1 RGB --feature_map_2 CORF3D --dataset_1 [path to dataset 1] --dataset_2 [path to dataset 2] --num_epochs 15 --resize 224 --batch_size 32 --classes 383 --trainable True --include_top False --learning_rate 0.0005 --model densenet121 
```

## Performance

#### Average performance of the best results achieved across leave-one-day-out and 5-fold cross-validation

| __Model__             | __Feature set__  | __Test accuracy__ (\%) | __F1__ (\%)          | __Method__ |
|:--------------------------:|:---------------------:|:---------------------------:|:-------------------------:|:-------------------------:|
| __DenseNet121 + SVM__      | __RGB + CORF3D__      | __99.64 ± 0.13__        | __99.54 ± 0.16__      | 5-Fold |
| __DenseNet121 + SVM__      | __RGB + CORF3D__      | __99.71 ± 0.30__        | __99.68 ± 0.30__      | Leave-one-day-out |

## Acknowledgment

We thank the Dairy Campus in Leeuwarden for permitting the data collection used in this project and for approving its availability for academic use. We are also grateful for the discussions held with Prof. Michael Biehl from the University of Groningen who contributed in the designing stage of several aspects of the project. Finally, we thank the Center for Information Technology of the University of Groningen for their support and for providing access to the Peregrine high performance computing cluster.

## Citation

If you find this paper or code useful, we encourage you to cite the paper. BibTeX:

      @article{BHOLE2021116354,
      title = {CORF3D contour maps with application to Holstein cattle recognition from RGB and thermal images},
      journal = {Expert Systems with Applications},
      pages = {116354},
      year = {2021},
      issn = {0957-4174},
      doi = {https://doi.org/10.1016/j.eswa.2021.116354},
      url = {https://www.sciencedirect.com/science/article/pii/S0957417421016511},
      author = {Amey Bhole and Sandeep S. Udmale and Owen Falzon and George Azzopardi},
      keywords = {Animal biometrics, Cattle recognition, Contour detection, ConvNets, Push-pull inhibition, Thermal images},
      abstract = {Livestock management involves the monitoring of farm animals by tracking certain physiological and phenotypical characteristics over time. In the dairy industry, for instance, cattle are typically equipped with RFID ear tags. The corresponding data (e.g. milk properties) can then be automatically assigned to the respective cow when they enter the milking station. In order to move towards a more scalable, affordable, and welfare-friendly approach, automatic non-invasive solutions are more desirable. Thus, a non-invasive approach is proposed in this paper for the automatic identification of individual Holstein cattle from the side view while exiting a milking station. It considers input images from a thermal-RGB camera. The thermal images are used to delineate the cow from the background. Subsequently, any occluding rods from the milking station are removed and inpainted with the fast marching algorithm. Then, it extracts the RGB map of the segmented cattle along with a novel CORF3D contour map. The latter contains three contour maps extracted by the Combination of Receptive Fields (CORF) model with different strengths of push-pull inhibition. This mechanism suppresses noise in the form of grain type texture. The effectiveness of the proposed approach is demonstrated by means of experiments using a 5-fold and a leave-one day-out cross-validation on a new data set of 3694 images of 383 cows collected from the Dairy Campus in Leeuwarden (the Netherlands) over 9 days. In particular, when combining RGB and CORF3D maps by late fusion, an average accuracy of 99.64%(±0.13) was obtained for the 5-fold cross validation and 99.71%(±0.31) for the leave–one day–out experiment. The two maps were combined by first learning two ConvNet classification models, one for each type of map. The feature vectors in the two FC layers obtained from training images were then concatenated and used to learn a linear SVM classification model. In principle, the proposed approach with the novel CORF3D contour maps is suitable for various image classification applications, especially where grain type texture is a confounding variable.}
      }
      
      @article{azzopardi2014push,
      title={A push-pull CORF model of a simple cell with antiphase inhibition improves SNR and contour detection},
      author={Azzopardi, George and Rodr{\'\i}guez-S{\'a}nchez, Antonio and Piater, Justus and Petkov, Nicolai},
      journal={PLoS One},
      volume={9},
      number={7},
      pages={e98424},
      year={2014},
      publisher={Public Library of Science San Francisco, USA}
    }

