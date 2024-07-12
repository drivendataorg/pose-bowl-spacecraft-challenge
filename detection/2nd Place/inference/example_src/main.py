import os
from pathlib import Path

import click
import pandas as pd
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np

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
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    assert (
        output_path.parent.exists()
    ), f"Expected output directory {output_path.parent} does not exist"

    logger.info(f"using data dir: {data_dir}")
    assert data_dir.exists(), f"Data directory does not exist: {data_dir}"

    # read in the submission format
    submission_format_path = data_dir / "submission_format.csv"
    submission_format_df = pd.read_csv(submission_format_path, index_col="image_id")

    # copy over the submission format so we can overwrite placeholders with predictions
    submission_df = submission_format_df.copy()

    model = YOLO("assets/best_openvino_model")

    image_dir = data_dir / "images"

    # add a progress bar using tqdm without spamming the log
    update_iters = min(100, int(submission_format_df.shape[0] / 10))
    with open(os.devnull, "w") as devnull:
        progress_bar = tqdm(
            enumerate(submission_format_df.index.values),
            total=submission_format_df.shape[0],
            miniters=update_iters,
            file=devnull,
        )
        for i, image_id in progress_bar:
            if (i % update_iters) == 0:
                logger.info(str(progress_bar))
            image_path = image_dir / f"{image_id}.png"
            assert image_path.exists(), f"Expected image not found: {image_path}"
            # load the image
            det = model(image_path, imgsz=1280, conf=0.1, iou=0.45, device="cpu", verbose=False)[0]
            boxes = det.boxes.xyxy.data.cpu().numpy()
            height, width = det.orig_shape[0:2]

            if boxes.shape[0] > 0:
                predicted = boxes[0,:].astype(int)
                sc_w, sc_h = predicted[2] - predicted[0], predicted[3] - predicted[1]
                ratio = max(sc_w/width, sc_h/height)
                if ratio > 0.7:
                    pad = 150
                    image = det.orig_img[:,:,::-1]
                    image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0) 
                    det = model(image, imgsz=1280, conf=0.1, iou=0.45, device="cuda", verbose=False)[0]
                    boxes = det.boxes.xyxy.data.cpu().numpy()
                    if boxes.shape[0] > 0:
                        boxes = boxes-pad

            if boxes.shape[0] == 0:
                det = model(image_path, imgsz=1280, conf=0.01, iou=0.45, device="cpu", verbose=False)[0]
                boxes = det.boxes.xyxy.data.cpu().numpy()

            if boxes.shape[0] == 0:
                xmin, ymin, xmax, ymax = [0, 0, width, height]
            else:
                xmin, ymin, xmax, ymax = boxes[0,:].astype(int)
                xmin = min(max(0, xmin), width)
                xmax = min(max(0, xmax), width)
                ymin = min(max(0, ymin), height)
                ymax = min(max(0, ymax), height)

            submission_df.loc[image_id] = pd.Series({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            })

    submission_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
