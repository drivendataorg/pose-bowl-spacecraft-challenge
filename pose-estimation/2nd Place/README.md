# 2nd Place Solution — Pose Bowl: Pose Estimation Track

<img src="assets/pose-estimate-tile_s.png" alt="spacecraft"  width=400/>

<small>Image courtesy of NASA/NTL</small>

## General

* **Competition Purpose:** Estimate satellite pose from chaser image.
* **Type:** Satellite Pose Estimation
* **Host:**  NASA Tournament Lab
* **Platform:** DrivenData
* **Competition link:** https://www.drivendata.org/competitions/261/spacecraft-pose-estimation/
* **Placement:** 2nd
* **User Name:** Ouranos
* **System used:** Local GPU Server
* **Solution documentation:** [write-up](./TKTKTK.pdf)
* **Train duration:** About 4 days
* **Inference duration:** About half an hour using CPU for 18*100 images

<br />

**High Level solution description.** Used a small detection model (yolo8n) to detect the satellite (using dataset of detection Track). Around it's center a rectangular of 384 pixels side is cropped and then a Siamese model with EfficientNetB0 backbone is trained using both the first image and one more with corresponding targets. For current solution only the 1st (= x) out of 7 targets was estimated and all other was set to 0. As a post processing step, all predictions are a weighted average with the previous in series prediction (`pred=0.9*pred + 0.1*previous_pred`) for every satellite series. Finally all predictions are clipped between (0, 400) and powered with 1.02 (`preds=np.clip(preds,0,400)**1.02`).


<br />


### System
| Characteristic   | Main Server      |
|------------------|------------------|
| Operating System | Ubuntu 18.04     |
| Cuda             | 11.2             |
| Python           | 3.8              |
| Pytorch          | 1.8              |
| GPU              | GeForce GTX 1080 |
| RAM              | 70GB             |

## Instructions

To retrain the models, the three notebooks numbered 0 through 2 should be run in order as described in the following sections.

### Data setup

This solution uses training data from both tracks of the challenge.

1. Download object detection training data from the [Detection Track](https://www.drivendata.org/competitions/260/spacecraft-detection/) into:
    - `datasets/detection/train_labels.csv`
    - `datasets/detection/train_metadata.csv`
    - `datasets/detection/images/`
2. Download pose estimation training data from the [Pose Estimation Track](https://www.drivendata.org/competitions/261/spacecraft-pose-estimation/)
    - `datasets/pose/train_labels.csv`
    - `datasets/pose/range.csv`
    - `datasets/pose/images/`

### Finding spacecraft centers

First finetune a yolo8n detection model (pytorch) on first Track competition data.

1. Set up a virtual environment (Python 3.8) for detection. Install the requirements with
    ```bash
    pip install -r requirements-yolo.txt
    ```
2. Run the detection model fine-tuning notebook: [`0-pose-bowl-detection-track_alldata_subm_v2.ipynb`](./0-pose-bowl-detection-track_alldata_subm_v2.ipynb).
3. Run the detection notebook to use the fine-tuned detection model to estimate the center of satellite in the pose estimation training data: [`1-siam_v19B2x_detect0_v2.ipynb`](./1-siam_v19B2x_detect0_v2.ipynb)

### Train Siamese model

Finally, train a keras Siamese model to estimate the first target (=x)

1. Set up a virtual environment (Python 3.8) for pose estimation. Install the requirements with
    ```bash
    pip install -r requirements-siam.txt
    ```
2. Run the Siamese model training notebook: [`siam_cc_v19B2x_v2.ipynb`](./2-siam_cc_v19B2x_v2.ipynb)

### Create submission

After running all of the notebooks, you can recreate the submission from the trained model artifacts. Copy the following files to the `submission/` folder:

- Fine-tuned detection model weights: `finetune/train/weights/last.pt` -> `submission/assets/last.pt`
- Siamese model weights: `weights/ep79siam_cc_v19B2xb.hdf5` -> `submission/assets/ep79siam_cc_v19B2xb.hdf5`

Then create a ZIP archive from the contents of the submission directory

```bash
(cd submission && zip -r ../submission.zip ./*)
```
