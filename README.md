[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://drivendata-prod-public.s3.amazonaws.com/comp_images/competition-group-tile-nasa-spacecraft.png' width='500'>](https://space-inspection.drivendata.org/)

# Pose Bowl: Spacecraft Detection and Pose Estimation Challenge

## Goal of the Competition

Inspector spacecraft, like NASA’s Seeker, are designed to conduct low-cost in-space inspections of other ships. Inspector spacecraft have limited computing resources, but complex computing demands.

In this challenge, solvers helped NASA develop algorithms that could be run on inspector (chaser) spacecraft. There were two tracks, with different associated prizes.

This challenge had two tracks:

1. **Detection Track**—develop object detection solutions that identify a bounding box around a spacecraft in an image
2. **Pose Estimation Track**—identify the relative position and orientation (pose) of the chaser spacecraft camera across sequences of images.

## What's in this Repository

This repository contains code from winning competitors in the [Pose Bowl: Spacecraft Detection and Pose Estimation Challenge](https://space-inspection.drivendata.org/) on DrivenData. Code for all winning solutions are open source under the MIT License.

Solution code for the two tracks can be found in the `detection/` and `pose-estimation/` subdirectories, respectively. Additional solution details can be found in the `reports` folder inside the directory for each submission.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

### Detection Track

| Place | Team or User  | Public Score | Private Score | Summary of Model                           |
|-------|---------------|--------------|---------------|--------------------------------------------|
| 1     | Polynome team | 0.9330       | 0.9261        | Uses iterative YOLOv8 object detection models. First, the object region is identified and the photo is cropped. Then, results are passed to a YOLOv8m refiner model trained on 300,000 synthetic images generated by a diffusion model with the supplemental no-background spacecraft model images.  |
| 2     | dungnb        | 0.9300       | 0.9226        | Uses a YOLOv8s 1280 model trained on over 100,000 images, including challenge data and synthetic images. The synthetic images varied spacecraft size, position, parts, and backgrounds, altering color, blur, brightness, and other details, using the supplemental no-background spacecraft images as well as a public dataset with additional spacecraft models. |
| 3     | agastya       | 0.9168       | 0.9141        | Uses a YOLOv8s 1280 model with manual hyperparameter tuning, and no post-processing. |

### Pose Estimation Track

| Place | Team or User       | Public Score | Private Score | Summary of Model                                                                                                                                                                                                                                                    |
|-------|--------------------|--------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | dylanliu           | 1.9312       | 1.9026        | Generates ORB and AKAZE features for each image and performs feature matching using OpenCV's Brute-Force matcher. Uses RANSAC to estimate homographic rotations as features. Relative pose is estimated using an ensemble of fine-tuned Chronos-T5 models using rotations and range values as features. |
| 2     | ouranos            | 1.9024       | 1.9311        | First uses a fine-tuned YOLOv8 object detection model to the center of the target spacecraft. Then uses a fine-tuned Siamese network with EfficientNetB0 backbone to estimate pose given a target image and the reference image. |
| 3     | OrbitSpinners team | 2.0016       | 1.9466        | Generates SIFT features for each image and performs feature matching using OpenCV's Brute-Force matcher. Using the matches, the relative pose is calculated using USAC-ACCURATE (graph-cut RANSAC variation). |

---

**Winners Blog Post: [https://drivendata.co/blog/posebowl-winners](https://drivendata.co/blog/posebowl-winners)**

**Benchmark Blog Post: [https://drivendata.co/blog/nasa-pose-bowl-object-detection-benchmark](https://drivendata.co/blog/nasa-pose-bowl-object-detection-benchmark)**
