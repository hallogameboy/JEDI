# JEDI for Circular RNA Prediction.

This repository provides the implementation of our paper: "JEDI: Circular RNA Prediction based on Junction Encoders and Deep Interaction among Splice Sites," Jyun-Yu Jiang, Chelsea J.-T. Ju, Junheng Hao, Muhao Chen and Wei Wang. (Submitted to ISMB'20).


## Required Packages

* Python 3.6.9 (or a compatible version)
* Tensorflow 2.0 (or a compatible version)
* NumPy 1.17.4 (or a compatible version)
* Abseil 1.13.0 (or a compatible version)
* scikit-learn 0.22 (or a compatible version)
* tqdm 4.40.1 (or a compatible version)
* Yaml 5.2 (or a compatible version)
* ujson (or be subsequently replaced with geuine json)


## Experimental Datasets and Settings

Please see the three datasets in [`data/`](data/) with detailed instructions.


## General Instructions for Conducting Experiments with JEDI

1. Prepare datasets for K-fold cross-validation following the designated JSON format as shown in the section of data preparation.
2. Execute the model with corresponding hyperparameters and experimental settings.
3. Done!

## Data Preparation

### Configuration

The file [`src/config.yml`](src/config.yml) is the configuration file in the yaml format for experiments as:
```
path_data: PATH_TO_PROCESSED_DATA_DIR
path_pred: PATH_TO_PREDICTION_DIR
```

* `path_data` represents the directory contatining processed data
* `path_pred` is the location, where JEDI will outupt the predictions for testing data.

This file is the default setting for JEDI in our implementation while the model supports to use an arbitrary configuration for conducting experiments. 

### Data Location

All of the data used by `path_data` should be put at `path_data` identified in the configuration file.

### Data Naming

Training and testing data should follow the desinated naming as `data.[Fold].K[K].L[L].[train/test]`, where 
* `[Fold]` is the fold number in cross-validation.
* `[K]` represents the size of *k*-mers for junction encoder.
* `[L]` is the length of flanking regions around junctions for modeling splice sites.
* `[train/test]` indicates the file belongs to either training or testing data.

For example, a file `data.0.K3.L4.train` is the training data for JEDI with 3-mers and length-4 flanking regions in the first fold in cross-validation. Note that the fold number can be any integer that is recognized later by command line arguments.

### Data Format

Each line in every training/testing file should match the following JSON format:

```
{
  "label": 0 or 1,
  "acceptors": [[L integers)], [L integers], ...],
  "donors": [[L integers)], [L integers], ...],
}  
```

* **"label"** maps to the label for this instnace, either 0 or 1.
* **"acceptors"** maps to a list of integer lists with *L* integers, where the *j*-th integer in the *i*-th list is the ID of the *j*-th *k*-mer in the flanking region of the *i*-th acceptor.
* **"donors"** maps to a list of integer lists with *L* integers, where the *j*-th integer in the *i*-th list is the ID of the *j*-th *k*-mer in the flanking region of the *i*-th donors.

Note that the *k*-mer IDs should start from 1 and cannot exceed or equal to 5<sup>*k*</sup> for the embedding purpose.

## Training, Testing, and Evaluation


### Hyper-parameters and experimental setttings through command line options.

### Training 

### Testing

### Evaluation


