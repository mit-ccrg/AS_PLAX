# AS_PLAX
This repo contains code and model for paper Identifying Aortic Stenosis with a single PLAX video

## Trained Model Weights
The trained model check points are shared in the dropbox folder:
https://www.dropbox.com/sh/vg0avrsrkkw4ipl/AADkZAZxW73VdxJ6gZG_OXQ0a?dl=0

MA: `./ma_checkpoint`

MP: `./mp_checkpoint`

MS: `./ms_checkpoint`

## Preprocessing of Dicom Files
The preprocessing include extracting pixel files from the raw Dicom files, de-identification, remove ECG and other meta data and resize the frames of the video. Source codes are under `./utils`

## Create a Dataset
To load a new dataset for training and inference, create a dataset file under `./datasets` like `dataset_example.py`.

## Training Model

To train a model, create a new config file under ./configs like `train_example.json`.

In this config file, define training configerations like path to data, label, hyperparamters and logging options, etc.

Command to launch training: `python train.py --json_file path/to/json`

## Testing Model

To test a model, create a new config file under ./configs like `test_example.json`.

In this config file, define testing configerations like path to data, ground truth label, model checkpoint and logging options, etc.

Command to launch testing: `python test.py --json_file path/to/json`