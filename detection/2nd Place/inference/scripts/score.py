import argparse
import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

INDEX_COLS = ["image_id"]
PREDICTION_COLS = ["xmin", "ymin", "xmax", "ymax"]


def _check_columns_as_expected(df: pd.DataFrame):
    assert len(PREDICTION_COLS) == len(
        df.columns
    ), f"Wrong number of columns, expected {len(PREDICTION_COLS)}"
    for i, col in enumerate(PREDICTION_COLS):
        assert (
                df.columns[i] == col
        ), f"{i}th column '{df.columns[i]}' does not match expected '{col}'"


def jaccard_index(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Calculate the Jaccard index (AKA intersection over union) for a set predictions
    compared to the ground truth.

    The input array columns should correspond to the bounding box coordinates:
    ["xmin", "ymin", "xmax", "ymax"]

    Args:
        predicted (np.ndarray): Nx4 array of predicted bounding boxes
        actual (np.ndarray): Nx4 array of ground truth bounding boxes

    Returns:
        np.ndarray: array of IoU scores, one per row of inputs
    """
    if actual.shape[1] != 4:
        raise ValueError("Predicted values must have 4 columns: xmin, ymin, xmax, ymax")
    missing = np.isnan(predicted).any().any()
    nonfinite = not np.isfinite(predicted).all().all()
    negative = (predicted < 0).any().any()
    if missing or nonfinite or negative:
        raise ValueError(
            "Predicted values must all be present, finite, and non-negative"
        )
    if (predicted[:, 0] > predicted[:, 2]).any():
        raise ValueError("All xmax must be greater than or equal to xmin")
    if (predicted[:, 1] > predicted[:, 3]).any():
        raise ValueError("All ymax must be greater than or equal to ymin")

    xmin = np.maximum(actual[:, 0], predicted[:, 0])
    xmax = np.minimum(actual[:, 2], predicted[:, 2])
    ymin = np.maximum(actual[:, 1], predicted[:, 1])
    ymax = np.minimum(actual[:, 3], predicted[:, 3])
    pred_height = predicted[:, 3] - predicted[:, 1]
    pred_width = predicted[:, 2] - predicted[:, 0]
    actual_height = actual[:, 3] - actual[:, 1]
    actual_width = actual[:, 2] - actual[:, 0]
    intersection_height = np.maximum(ymax - ymin, 0)
    intersection_width = np.maximum(xmax - xmin, 0)
    if not ((actual_height * actual_width) > 0).all():
        raise ValueError("Not all rows of ground truth dataset have a non-zero area")

    area_of_intersection = intersection_height * intersection_width
    area_of_union = (
            actual_height * actual_width + pred_height * pred_width - area_of_intersection
    )
    return area_of_intersection / area_of_union


def score_rows(predicted_df: pd.DataFrame, actual_df: pd.DataFrame) -> Dict[str, float]:
    """Scores a set of predicted object bounding boxes against the ground truth. Returns the
    micro-average of the Jaccard index (AKA the intersection over union).

    Args:
        predicted_df (np.ndarray): dataframe of predicted bounding boxes
        actual_df (np.ndarray): dataframe of ground truth bounding boxes

    Returns:
        Dict[str, float]: micro-averaged IoU
    """
    iou_scores = jaccard_index(predicted_df.values, actual_df.values)
    return {"score": np.mean(iou_scores)}


def main(
        predicted_path: Union[str, Path], actual_path: Union[str, Path]
) -> Dict[str, float]:
    """Calculates the Jaccard index score for the Pose Bowl: Spacecraft Detection and Pose Estimation Challenge.

    Args:
        predicted_path (str | Path): Path to predictions CSV file matching submission format
        actual_path (str | Path): Path to ground truth CSV file

    Returns:
        Dict[str, float]: Jaccard index score
    """
    try:
        predicted_df = pd.read_csv(predicted_path, index_col=INDEX_COLS)
        _check_columns_as_expected(predicted_df)
    except ValueError as e:
        raise ValueError("Submitted index column is missing.") from e
    except (IndexError, AssertionError) as e:
        raise ValueError(
            f"Submitted columns do not match expected columns {INDEX_COLS + PREDICTION_COLS}"
        ) from e
    try:
        actual_df = pd.read_csv(actual_path, index_col=INDEX_COLS)
        _check_columns_as_expected(actual_df)
    except ValueError as e:
        raise ValueError("Ground truth index columns are missing.") from e
    except (IndexError, AssertionError) as e:
        raise ValueError(
            f"Submitted columns do not match expected columns {INDEX_COLS + PREDICTION_COLS}"
        ) from e

    if not predicted_df.index.equals(actual_df.index):
        raise ValueError("Submitted index does not match expected index")

    return score_rows(predicted_df=predicted_df, actual_df=actual_df)


parser = argparse.ArgumentParser(description=main.__doc__)
parser.add_argument("predicted_path", help="Path to predictions CSV.")
parser.add_argument("actual_path", help="Path to ground truth CSV.")

if __name__ == "__main__":
    args = parser.parse_args()
    result_dict = main(predicted_path=args.predicted_path, actual_path=args.actual_path)
    print(json.dumps(result_dict, indent=2))
