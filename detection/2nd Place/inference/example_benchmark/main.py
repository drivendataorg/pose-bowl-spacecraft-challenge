import os
from pathlib import Path

import click
import cv2
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO

INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def centered_box(img, scale=0.1):
    """
    Return coordinates for a centered bounding box on the image, defaulting to 10% of the image's height and width.
    """
    # Get image dimensions
    height, width, _ = img.shape
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2
    # Calculate 10% of the image's height and width for the bounding box
    box_width, box_height = width * scale, height * scale
    # Calculate top-left corner of the bounding box
    x1 = center_x - box_width // 2
    y1 = center_y - box_height // 2
    # Calculate bottom-right corner of the bounding box
    x2 = center_x + box_width // 2
    y2 = center_y + box_height // 2

    return [x1, y1, x2, y2]


@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False),
)
def main(data_dir, output_path):
    # locate key files and locations
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    submission_format_path = data_dir / "submission_format.csv"
    images_dir = data_dir / "images"

    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"
    assert output_path.parent.exists(), f"Expected output directory {output_path.parent} does not exist"
    assert submission_format_path.exists(), f"Expected submission format file {submission_format_path} does not exist"
    assert images_dir.exists(), f"Expected images dir {images_dir} does not exist"
    logger.info(f"using data dir: {data_dir}")

    # copy the submission format file; we'll use this as template and overwrite placeholders with our own predictions
    submission_format_df = pd.read_csv(submission_format_path, index_col="image_id")
    submission_df = submission_format_df.copy()
    # load pretrained model we included in our submission.zip
    model = YOLO('yolov8n.pt')
    # add a progress bar using tqdm without spamming the log
    update_iters = min(100, int(submission_format_df.shape[0] / 10))
    with open(os.devnull, "w") as devnull:
        progress_bar = tqdm(
            enumerate(submission_format_df.index.values),
            total=submission_format_df.shape[0],
            miniters=update_iters,
            file=devnull,
        )
        # generate predictions for each image
        for i, image_id in progress_bar:
            if (i % update_iters) == 0:
                logger.info(str(progress_bar))
            # load the image
            img = cv2.imread(str(images_dir / f"{image_id}.png"))
            # get yolo result
            result = model(img, verbose=False)[0]
            # get bbox coordinates if they exist, otherwise just get a generic box in center of an image
            bbox = result.boxes.xyxy[0].tolist() if len(result.boxes) > 0 else centered_box(img)
            # convert bbox values to integers
            bbox = [int(x) for x in bbox]
            # store the result
            submission_df.loc[image_id] = bbox
    # write the submission to the submission output path
    submission_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
