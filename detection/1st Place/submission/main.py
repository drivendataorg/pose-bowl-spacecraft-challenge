import os
from pathlib import Path

import click
import cv2
import pandas as pd
from loguru import logger
from tqdm import tqdm
from ultralytics import YOLO


def centered_box(img, scale=0.1):
    """Return coordinates for a centered bounding box on the image, defaulting to 10% of the image's height and width."""
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


def expand_region(box, w, h, scale=0.75):
    ww = box[2] - box[0]
    hh = box[3] - box[1]
    x1 = int(max(0, box[0] - scale * ww))
    y1 = int(max(0, box[1] - scale * hh))
    x2 = int(min(w, box[2] + scale * ww))
    y2 = int(min(h, box[3] + scale * hh))
    return [x1, y1, x2, y2]


def cal_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    x3 = min(b1[0], b2[0])
    y3 = min(b1[1], b2[1])
    x4 = max(b1[2], b2[2])
    y4 = max(b1[3], b2[3])
    return ((x2 - x1) * (y2 - y1)) / ((x4 - x3) * (y4 - y3))


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
    model3 = YOLO(
        "./best_int8_openvino_model_m"
    )  # ('./submission906/best_openvino_model_1280s')#.to(torch.device("cpu"))
    model = YOLO("./best_int8_openvino_model_n")
    model2 = YOLO("./best_openvino_model_640")

    model4 = YOLO("./best_int8_openvino_model_s")
    # model3 = YOLO('./best_openvino_model_2048')
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
            _anybox = None
            flag = False
            if (i % update_iters) == 0:
                logger.info(str(progress_bar))
            # load the image
            img = cv2.imread(str(images_dir / f"{image_id}.png"))

            h, w, _ = img.shape

            # get yolo result
            result = model(img, verbose=False, imgsz=1280, conf=0.3)[0]
            if len(result.boxes) > 0:
                _b = result.boxes.xyxy[0].tolist()
            # get bbox coordinates if they exist, otherwise just get a generic box in center of an image
            if len(result.boxes) > 0:
                init_bbox = result.boxes.xyxy[0].tolist()
                _anybox = init_bbox.copy()
                scale = 0.75
                init_bbox = expand_region(init_bbox, w, h, scale=scale)
                new_img = img[init_bbox[1] : init_bbox[3], init_bbox[0] : init_bbox[2], :]
                result2 = model2(new_img, verbose=False, imgsz=640, conf=0.1)[0]
                if len(result2.boxes) > 0:
                    bbox = result2.boxes.xyxy[0].tolist()
                    bbox[0] += init_bbox[0]
                    bbox[1] += init_bbox[1]
                    bbox[2] += init_bbox[0]
                    bbox[3] += init_bbox[1]
                else:
                    flag = True

            else:
                flag = True

            if flag:
                flag = False
                result = model4(img, verbose=False, imgsz=1280, conf=0.3)[0]
                # # get bbox coordinates if they exist, otherwise just get a generic box in center of an image
                if len(result.boxes) > 0:
                    init_bbox = result.boxes.xyxy[0].tolist()
                    _anybox = init_bbox.copy()
                    if result.boxes.conf[0] > 0.9:
                        bbox = init_bbox.copy()
                    else:
                        scale = 0.75
                        init_bbox = expand_region(init_bbox, w, h)
                        new_img = img[init_bbox[1] : init_bbox[3], init_bbox[0] : init_bbox[2], :]
                        result2 = model2(new_img, verbose=False, imgsz=640, conf=0.1)[0]
                        result3 = model2(cv2.flip(new_img, 1), verbose=False, imgsz=640, conf=0.1)[0]
                        if len(result2.boxes) > 0:
                            bbox = result2.boxes.xyxy[0].tolist()
                            conf = result2.boxes.conf[0]
                            if len(result3.boxes) > 0:
                                bbox2 = result3.boxes.xyxy[0].tolist()
                                conf2 = result3.boxes.conf[0]
                                temp = bbox2[0]
                                bbox2[0] = new_img.shape[1] - bbox2[2]
                                bbox2[2] = new_img.shape[1] - temp
                            else:
                                bbox2 = bbox
                                conf2 = conf
                            a1 = conf / (conf + conf2)
                            a2 = conf2 / (conf + conf2)
                            b1 = bbox
                            b2 = bbox2
                            newx1 = a1 * b1[0] + a2 * b2[0]
                            newy1 = a1 * b1[1] + a2 * b2[1]
                            newx2 = a1 * b1[2] + a2 * b2[2]
                            newy2 = a1 * b1[3] + a2 * b2[3]
                            bbox = [newx1, newy1, newx2, newy2]
                            bbox[0] += init_bbox[0]
                            bbox[1] += init_bbox[1]
                            bbox[2] += init_bbox[0]
                            bbox[3] += init_bbox[1]
                        else:
                            flag = True

                else:
                    flag = True

            if flag:
                result = model3(img, verbose=False, imgsz=1280, conf=0.01)[0]
                if len(result.boxes) > 0:
                    init_bbox = result.boxes.xyxy[0].tolist()
                    if result.boxes.conf[0] > 0.5:
                        bbox = init_bbox.copy()
                    else:
                        init_bbox = expand_region(init_bbox, w, h)
                        new_img = img[init_bbox[1] : init_bbox[3], init_bbox[0] : init_bbox[2], :]
                        result = model2(new_img, verbose=False, imgsz=640, conf=0.1)[0]
                        if len(result.boxes) > 0:
                            bbox = result.boxes.xyxy[0].tolist()
                            bbox[0] += init_bbox[0]
                            bbox[1] += init_bbox[1]
                            bbox[2] += init_bbox[0]
                            bbox[3] += init_bbox[1]
                        else:
                            bbox = init_bbox
                else:
                    bbox = centered_box(img)

            # convert bbox values to integers
            bbox = [int(x) for x in bbox]
            # store the result
            submission_df.loc[image_id] = bbox
    # write the submission to the submission output path
    submission_df.to_csv(output_path, index=True)


if __name__ == "__main__":
    main()
