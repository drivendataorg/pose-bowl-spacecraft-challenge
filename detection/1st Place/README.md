# Pose_Bowl_detection_track (First place solution)

This is the official implementation of the winning solution for the Detection Track in the
Pose Bowl competition.

## Setup the environment
First we can start by using the following docker container for the training

```
skypirate91/minigpt4:0.4
```

If you want to use the same hardware we used please refer to the report hardware section.

## Training the detectors
All the scripts and notebooks are located inside the `scripts/Detectors` directory

### Data preparation
We provide a notebook (`scripts/Detectors/Data_splitting_pre.ipynb`) to convert the competition
data to YOLO format.  As well, we show how we splitted the data to tune the hyperparameter.
The final training was on the full dataset on all experiments.

### Training
After data preparation pay attention that the paths are correct on the `scripts/Detectors/na_all.yaml`
config file and you can start the training.

For example to train our nano detector run:
```
bash nano.sh
```
To reproduce the results you need to train also the small and medium versions
```
bash small.sh
bash medium.sh
```

### Quantization

To quantize the models you need to run the quantization script and pass the best checkpoint you
got from the training for example to quantize the nano model

```
bash quantize.sh
```

Please pay attention that for full reproducability you need to use different ultralytics
version when quantizing the `scripts/Detectors/quantize.sh` small version. 

## Training the refiner
All the scripts and notebooks are located inside the scripts/Refiner directory

### Data preparation

#### Data Generation
This step will take a while and it is important to pay attention to the paths.
We will generate data using [Kandinsky3](https://huggingface.co/kandinsky-community/kandinsky-3)
as background and then inject the no-background spaceships in these generated data to get a synthetic
dataset of 300K images.

This work is done on this `scripts/Refiner/Gen_data_300k.ipynb` notebook.

Pay attention that you need to restart the notebook after generating the background to
avoid any randomness in inserting the spaceships into the generated data. at this point it
is not important where you store the generated data just save the path to pass it to the cropping stage

#### Synthetic data cropping
At this step we will crop the region around the spaceship in the synthetic data randomly
using the `scripts/Refiner/crop_data_syn.ipynb` notebook.

Adjust the paths for the generated data to follow yolo format (images and labels folders) and
use the synthetic data path from the previous step for reading.

Add the path of the cropped data to the `scripts/Refiner/na_all_crop_syn3.yaml` config file;
This data will be used to train the refiner stage 1.

#### Original data cropping

Similar to the previous step but to be applied on the original data instead,
you can use this `scripts/Refiner/crop_data.ipynb` notebook to adjust the paths.

Add the path of the cropped data to the `scripts/Refiner/na_all_crop.yaml` config file.
This data will be used to train the refiner stage 2.

### Refiner training
Just run:
```
bash refiner_stage1.sh
```
add the path of the last checkpoint of stage 1 in `scripts/Refiner/refiner_stage2.sh`.
Then run:

```
bash refiner_stage2.sh
```

### Export to OpenVino

You need to edit the model path in the script `scripts/Refiner/openvino.sh`, then run: 

```
bash openvino.sh
```

## Inference

Before submitting, two library wheels will need to be placed in the submission folder:

```
python -m pip download --no-deps --dest submission/ --no-cache openvino==2024.0.0
python -m pip download --no-deps --dest submission/ --no-cache ultralytics==8.1.11
```