# Pose Estimation Track — 3rd Place Solution

This is the solution by team OrbitSpinnersChallengeWinners for the Pose Estimation Track for the Pose Bowl Spacecraft Detection and Pose Estimation Challenge.

* Stepan Konev
* Yuriy Biktairov

---

This solution uses the challenge's inference runtime container and code provided by DrivenData at [drivendataorg/spacecraft-pose-pose-estimation-runtime](https://github.com/drivendataorg/spacecraft-pose-pose-estimation-runtime). The source code specific to this solution can be found in the [`example_src/`](./example_src/) directory.

The required code is self-contained. The containerised structure allows for a great simplification for running and testing the solution. To run inference one needs to run:

```bash
# To pull the prebuilt Docker image
make pull

# To instead build the image locally
# make build

make clean && make pack-example
make test-submission
```

Please make sure to have prerequisites installed and data properly located as described in the original instructions below at [Quickstart](#quickstart) section.

This code was run on CPU only with:

* Ubuntu 22.04 LTS
* AMD Ryzen Threadripper Pro
* For offline testing we limited the available RAM and number of CPUs to 4GB and 3 cores.

To obtain the final results the code was inferenced on the remote testing server. The solution does not require the
training phase and is ready for inference.

### Solution source code

The structure of the code is the following:

The code for the solution is contained in [`example_src/`](./example_src/) folder. `example_src/main.py` generally handles data loading chain by chain and passing it to the prediction module. I also implemented the logic for compiling the submission file.

`example_src/localizator.py` contains prediction-related logic and a few helper functions. The main function there is `predict_chain` which implements sequential processing of the chain images. Lines 81-111 implement the general logic fetching the corresponding matches between the images and utilizing them for pose estimation. Lines 112 - 117 implement a fallback heuristic for case when the pose estimation algorithm failed. It might be a case if the pose has dramatically changes between the two states and thus the matcher is unable to fetch the matches.

---

> [!NOTE]
> The following documentation is a modified copy of the "Quickstart" instructions from the README in [drivendataorg/spacecraft-pose-pose-estimation-runtime](https://github.com/drivendataorg/spacecraft-pose-pose-estimation-runtime). Credit for the original documentation belongs to DrivenData.
>
> The contents of `runtime/`, `scripts/`, and `Makefile` are also copied from that repository. See [LICENSE-THIRD-PARTY](./LICENSE-THIRD-PARTY).

## Quickstart

This section guides you through the steps to test a simple but valid submission for the competition.

### Prerequisites

First, make sure you have the prerequisites installed.

 - A clone or fork of this repository
 - Enough free space on your machine for the spacecraft images dataset (at least 10 GB) and Docker container images (5 GB)
 - [Docker](https://docs.docker.com/get-docker/)
 - [GNU make](https://www.gnu.org/software/make/) (optional, but useful for running commands in the Makefile)

### Download the data

First, go to the challenge [download page](https://www.drivendata.org/competitions/261/spacecraft-pose-estimation/data/) to start downloading the challenge data. Save the `submission_format.csv` and `training_labels.csv` in this project's `/data` directory.

The images dataset is broken up into individual tar files of approximately 10 GB in size. Download at least one of these tar files to get started, and then extract it to the `data/images` directory.

Once everything is downloaded and in the right location, it should look something like this:

```
data/                         # Runtime data directory
├── images/                   # Directory containing image files
│      │
│      ├── a0a0d73d0e9a4b16a23bc210a264fd3f.png
│      ├── a0a6efb87e1fcd0c158ba35ced823250.png
│      ├── a0a0d73d0e9a4b16a23bc210a264fd3f.png
│      ├── a0a6efb87e1fcd0c158ba35ced823250.png
│      └── ...
│
├── submission_format.csv     # CSV file showing how submission should be formatted
└── train_labels.csv          # CSV file with ground truth data
```

Later in this guide, when we launch a Docker container from your computer (or the "host" machine), the `data` directory on your host machine will be mounted as a read-only directory in the container as `/code_execution/data`. In the runtime, your code will then be able to access all the competition data at `/code_execution/data`, which will by default look to your script like `./data` since your script will be invoked with `/code_execution` as the working directory.

### The quickstart example

> [!IMPORTANT]
> The source code from the [original quickstart example](https://github.com/drivendataorg/spacecraft-pose-pose-estimation-runtime/tree/main/example_src) has been replaced with the source code for this solution.
>
> `main.sh` is the main entrypoint that is run when running the container.

### Testing the submission

The primary purpose of this runtime repository is to allow you to easily test your submission before making a submission to the DrivenData platform.

Your submission is going to run inside a Docker container on our code execution platform. This repository contains the definition for that (Docker container)[https://github.com/drivendataorg/spacecraft-pose-pose-estimation-runtime/tree/main/runtime], as well as a few commands you can run to easily download the Docker image and test your submission. Below we walk through those commands.

First, make sure Docker is running and then run the following commands in your terminal:

1. **`make pull`** downloads the latest official Docker image from the container registry ([Azure](https://azure.microsoft.com/en-us/services/container-registry/)). You'll need an internet connection for this.
2. **`make pack-example`** zips the contents of the `example_src` directory and saves it as `submission/submission.zip`. This is the file that you will upload to the DrivenData competition site for code execution. But first we'll test that everything looks good locally in step #3.
   * Note: When running this again in the future, you may need to first run `make clean` before you re-pack the example for submission, both because it won't rerun by default if the submission file already exists, and also because sometimes running with Docker before may have created files in the mounted submission directory with different permissions.
3. **`make test-submission`** will do a test run of your submission, simulating what happens during actual code execution. This command runs the Docker container with the requisite host directories mounted, and executes `main.sh` to produce a CSV file with your image rankings at `submission/submission.csv`.

```sh
make pull
make clean && make pack-example
make test-submission
```

🎉 **Congratulations!** You've just tested a submission for the Pose Bowl challenge. If everything worked as expected, you should see a new file has been generated at `submission/submission.csv`.

### Evaluating locally

In your local model development and cross validation, you may wish to use the same scoring
metric that will be employed when your real submissions are scored. We have included a script
that implements the same logic at `scripts/score.py`.

The usage is:

```
❯ python scripts/score.py --help
usage: score.py [-h] predicted_path actual_path

Calculates the pose error score for the Pose Bowl: Spacecraft Detection and Pose Estimation Challenge. Args: predicted_path (str | Path): Path to
predictions CSV file matching submission format actual_path (str | Path): Path to ground truth CSV file Returns: dict[int, Dict[str, float]]:
Dictionary of scores for each trajectory

positional arguments:
  predicted_path  Path to predictions CSV.
  actual_path     Path to ground truth CSV.

options:
  -h, --help      show this help message and exit
```

For example, using the `submission_format.py` as the predictions with our training labels as the
ground truth, we can verify that we achieve a (bad!) score:

```
❯ python scripts/score.py data/submission_format.csv data/train_labels.csv
{
  "mean_translation_error": 1.0,
  "mean_rotation_error": 1.0,
  "score": 2.0
}
```
