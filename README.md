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

### File Naming

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

All of the expeirmental setups and model hyper-parameters can be assigned through the command line options of our implementation. To be specific, the definitions of all options are listed in the function `handle_flags()` in [`src/utils.py`](src/utils.py) as follows.

```
def handle_flags():
    flags.DEFINE_string("tflog", '3', "The setting for TF_CPP_MIN_LOG_LEVEL (default: 3)")
    # Data configuration.
    flags.DEFINE_string('config' ,'config.yml', 'configure file (default: config.yml)')
    flags.DEFINE_integer('cv',  0, 'Fold for cross-validation (default: 0)')
    flags.DEFINE_integer('K',   3, 'K for k-mers (default: 3)')
    flags.DEFINE_integer('L',   4, 'Length for junction sites (default: 4)')
    
    # Model parameters.
    flags.DEFINE_integer('emb_dim',    128, 'Dimensionality for k-mers (default: 12)')
    flags.DEFINE_integer('rnn_dim',    128, 'Dimensionality for RNN layers (default: 128)')
    flags.DEFINE_integer('att_dim',     16, 'Dimensionality for attention layers (default: 16)')
    flags.DEFINE_integer('hidden_dim', 128, 'Dimensionality for hidden layers (default: 128)')
    flags.DEFINE_integer('max_len',    128, 'Max site number for acceptors/donors (default: 128)')

    # Training parameters.
    flags.DEFINE_integer("batch_size",    64, "Batch Size (default: 64)")
    flags.DEFINE_integer("num_epochs",    10, "Number of training epochs (default: 10)")
    flags.DEFINE_integer('random_seed',  252, 'Random seeds for reproducibility (default: 252)')
    flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate while training (default: 1e-3)')
    flags.DEFINE_float('l2_reg',        1e-3, 'L2 regularization lambda (default: 1e-3)')
    FLAGS = flags.FLAGS
```

### Training and Testing with Evaluation

Both training and testing procedures can be achived by the script [`src/run.py`](src/run.py) with the above options. For example, to run JEDI with the experimental settings reported in the paper on the first fold in CV for 2 epochs, we can execution the following command:
```
$ python3 run.py --cv=0 --K=3 --L=4 --emb_dim=128 --rnn_dim=128 --att_dim=16 --hidden_dim=128  --num_epochs=2 --learning_rate=1e-3 --l2_reg=1e-3
```

For each epoch, the script will show the progress during training and show the training evaluation metrics, together with the training loss, after training the whole training set once. As an example, the above command will result in the following results:

```
1 Physical GPUs, 1 Logical GPU
I0131 XX:XX:XX.XXXXXX XXXXXXXXXXXXXXX utils.py:82] Loaded XXXXX records from /PATHTO/data.0.K3.L4.train.
I0131 XX:XX:XX.XXXXXX XXXXXXXXXXXXXXX utils.py:82] Loaded XXXXX records from /PATHTO/data.0.K3.L4.test.
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 646/646 [00:42<00:00, 15.20it/s]
Epoch 1 (CV=0, K=3, L=4)
Ls: 0.12653037905693054 A: 0.9476475288761895    P: 0.9459974829335266  F: 0.9582399752762112,  M: 0.8886609741180113   Se: 0.9708034910571015  Sp: 0.9100723993395148

Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 646/646 [00:28<00:00, 22.28it/s]
Epoch 2 (CV=0, K=3, L=4)
Ls: 0.03979344666004181 A: 0.9857858924377073    P: 0.9926197805667377  F: 0.9884650906875749,  M: 0.9700091674012169   Se: 0.9843450354193574  Sp: 0.988123967991871
```

After the assigned number of epochs, the model will be applied to predict the testing data and compute the testing evaluation metrics as:

```
Testing: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 162/162 [00:05<00:00, 29.09it/s]
Testing (CV=0, K=3, L=4)
Ls: 0.03267291560769081 A: 0.988861985472155     P: 0.9932379304922158  F: 0.9909782693967208,  M: 0.9764442824547588   Se: 0.9887288666249218  Sp: 0.9890779781559563
```



### Predictions

Finally, the script will also dump the testing predictions into the desinated folder for the potential of conducting further analysis as:

```
I0131 XX:XX:XX.XXXXXX XXXXXXXXXXXXXXX run.py:143] Saving testing predictions to to /PATHT/pred.0.K3.L4.
```

Note the format of the predictions are in the JSON format as:
```
[
  [
    prediction score for the testing example 0,
    label for  the testing example 0
  ],
  [
    prediction score for the testing example 1,
    label for  the testing example 1
  ],
  ...
]
```



