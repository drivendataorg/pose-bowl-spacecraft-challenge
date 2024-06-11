import torch
from torch import nn
import sklearn
from glob import glob
import cv2
import pickle
import random
import numpy as np
import pandas as pd
import os
import typing as tp
from tqdm import tqdm
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import argparse
import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import numpy.typing
import pandas as pd

class CFG_base():
    max_epoch = 450
    batch_size = 1
    weight_decay = 1e-4
    es_patience =  30000
    deterministic = True
    enable_amp = True
    device = "cuda"
    lag = 0
    nrots = 4
    used_nrots = 4
    n_feats = 2
    newfeat_lags = [1, 2, 3]
CFG = CFG_base()
MODEL_CONFIGS = []

CFG.model_name = 'model_t1'
CFG.decoder_name = "amazon/chronos-t5-tiny"
CFG.lr = 2e-4
CFG.seed = 59
CFG.fold = 0
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_b2'
CFG.decoder_name = "amazon/chronos-t5-base"
CFG.lr = 5e-5
CFG.seed = 10
CFG.fold = 1
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_b3'
CFG.decoder_name = "amazon/chronos-t5-base"
CFG.lr = 5e-5
CFG.seed = 10
CFG.fold = 2
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_t4'
CFG.decoder_name = "amazon/chronos-t5-tiny"
CFG.lr = 2e-4
CFG.seed = 59
CFG.fold = 3
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_t5'
CFG.decoder_name = "amazon/chronos-t5-tiny"
CFG.lr = 2e-4
CFG.seed = 59
CFG.fold = 4
MODEL_CONFIGS.append(copy.deepcopy(CFG))

CFG.model_name = 'model_m1'
CFG.decoder_name = "amazon/chronos-t5-mini"
CFG.lr = 1e-4
CFG.seed = 1039
CFG.fold = 0
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_m2'
CFG.decoder_name = "amazon/chronos-t5-mini"
CFG.lr = 1e-4
CFG.seed = 1039
CFG.fold = 1
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_m3'
CFG.decoder_name = "amazon/chronos-t5-mini"
CFG.lr = 1e-4
CFG.seed = 1039
CFG.fold = 2
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_m4'
CFG.decoder_name = "amazon/chronos-t5-mini"
CFG.lr = 1e-4
CFG.seed = 1039
CFG.fold = 3
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_m5'
CFG.decoder_name = "amazon/chronos-t5-mini"
CFG.lr = 1e-4
CFG.seed = 1039
CFG.fold = 4
MODEL_CONFIGS.append(copy.deepcopy(CFG))

CFG.model_name = 'model_s1'
CFG.decoder_name = "amazon/chronos-t5-small"
CFG.lr = 1e-4
CFG.seed = 103
CFG.fold = 0
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_s2'
CFG.decoder_name = "amazon/chronos-t5-small"
CFG.lr = 1e-4
CFG.seed = 103
CFG.fold = 1
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_s3'
CFG.decoder_name = "amazon/chronos-t5-small"
CFG.lr = 1e-4
CFG.seed = 103
CFG.fold = 2
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_s4'
CFG.decoder_name = "amazon/chronos-t5-small"
CFG.lr = 1e-4
CFG.seed = 103
CFG.fold = 3
MODEL_CONFIGS.append(copy.deepcopy(CFG))
CFG.model_name = 'model_s5'
CFG.decoder_name = "amazon/chronos-t5-small"
CFG.lr = 2e-4
CFG.seed = 5515
CFG.fold = 4
MODEL_CONFIGS.append(copy.deepcopy(CFG))

FOLDS = [
    ['c1', 'c12'], 
    ['c13', 'c17', 'c5'], 
    ['c0', 'c4', 'c9'],
    ['c10', 'c11', 'c2'],
    ['bc15', 'c16', 'c3', 'c8']
]


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def quaternion_inverse(q):
    """Calculate the inverse of a quaternion."""
    norm_squared = np.sum(q**2)
    inverse = np.array([q[0], -q[1], -q[2], -q[3]]) / norm_squared
    return inverse

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return np.array([w, x, y, z])

def apply_delta_quaternion(base_quaternion, delta_quaternion_angle):
    
    # Apply the delta quaternion to the base quaternion
    result_quaternion_angle = quaternion_multiply(delta_quaternion_angle, base_quaternion)
    
    return result_quaternion_angle

def rotate_point_by_quaternion(point, quaternion):
    p_quaternion = np.concatenate(([0], point))
    conjugate_q = np.array([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    rotated_p = quaternion_multiply(quaternion_multiply(quaternion, p_quaternion), conjugate_q)[1:]
    return rotated_p

def rotation_matrix_to_quaternion(rotation):
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    q = np.zeros(4)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (rotation[2][1] - rotation[1][2]) * s
        q[2] = (rotation[0][2] - rotation[2][0]) * s
        q[3] = (rotation[1][0] - rotation[0][1]) * s
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2])
        q[0] = (rotation[2][1] - rotation[1][2]) / s
        q[1] = 0.25 * s
        q[2] = (rotation[0][1] + rotation[1][0]) / s
        q[3] = (rotation[0][2] + rotation[2][0]) / s
    elif rotation[1][1] > rotation[2][2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2])
        q[0] = (rotation[0][2] - rotation[2][0]) / s
        q[1] = (rotation[0][1] + rotation[1][0]) / s
        q[2] = 0.25 * s
        q[3] = (rotation[1][2] + rotation[2][1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1])
        q[0] = (rotation[1][0] - rotation[0][1]) / s
        q[1] = (rotation[0][2] + rotation[2][0]) / s
        q[2] = (rotation[1][2] + rotation[2][1]) / s
        q[3] = 0.25 * s

    return q / np.linalg.norm(q)

#import quaternion

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion to rotation matrix.
    """
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rotation_matrix = np.stack([
        1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
    ], axis=1).reshape(-1, 3, 3)
    return rotation_matrix

def angle_between_quaternions(q1, q2):
    """
    Calculate angle between two quaternions.
    """
    dot_product = np.abs(np.sum(q1 * q2, axis=-1))
    if np.abs(dot_product) > 1:
        dot_product = 1
    angle = 2 * np.arccos(dot_product)
    return angle


TRANSLATION_COLS = ["x", "y", "z"]
QUATERNION_COLS = ["qw", "qx", "qy", "qz"]
POSE_COLS = TRANSLATION_COLS + QUATERNION_COLS
INDEX_COLS = ["chain_id", "i"]
REFERENCE_VALUES = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
REFERENCE_SERIES = pd.Series(REFERENCE_VALUES, index=POSE_COLS)
REFERENCE_TOLERANCE = 1e-6

_EPS = np.finfo(float).eps * 4.0

# ESA metric machine precision constants
# ref: https://kelvins.esa.int/pose-estimation-2021/scoring/
TRANS_ERR_THRESHOLD = 0.002173
ROT_ERR_THRESHOLD = 0.169 * np.pi / 180

# Our tweaks to ESA metric—cap the maximum value of the translation error (so that it's not unbounded)
TRANS_MAX_ERROR = 1e3


def transformation_matrix(t: np.ndarray, q: np.ndarray):
    """Generate 4x4 homogeneous transformation matrices, a.k.a. SE(3) matrices, from arrays of 3D point translations
    and unit quaternion representations of rotations. This function is vectorized and operates on arrays of
    transformation parameters.

    Args:
        t (np.ndarray): Nx3 array of translations (tx, ty, tz)
        q (np.ndarray): Nx4 array of unit quaternions (qw, qx, qy, qz)

    Returns:
        np.ndarray: Nx4x4 array of 4x4 transformation matrices
    """
    # Allocate transformation matrices with only translation
    arr_transforms = np.repeat(np.eye(4)[None, :, :], t.shape[0], axis=0)
    arr_transforms[:, :3, 3] = t

    # shift so we have (qx, qy, qz, qw) instead of (qw, qx, qy, qz)
    q = np.roll(q, -1, axis=1)

    nq = np.square(np.linalg.norm(q, axis=1))
    mask = nq >= _EPS  # mask for rotations of magnitude greater than epsilon

    # For transformations with non-zero rotation, calculate rotation matrix
    q = np.sqrt(2.0 / nq)[:, None] * q
    q = q[:, :, None] * q[:, None, :]  # outer product
    arr_transforms[mask, 0, 0] = 1.0 - q[mask, 1, 1] - q[mask, 2, 2]
    arr_transforms[mask, 0, 1] = q[mask, 0, 1] - q[mask, 2, 3]
    arr_transforms[mask, 0, 2] = q[mask, 0, 2] + q[mask, 1, 3]
    arr_transforms[mask, 1, 0] = q[mask, 0, 1] + q[mask, 2, 3]
    arr_transforms[mask, 1, 1] = 1.0 - q[mask, 0, 0] - q[mask, 2, 2]
    arr_transforms[mask, 1, 2] = q[mask, 1, 2] - q[mask, 0, 3]
    arr_transforms[mask, 2, 0] = q[mask, 0, 2] - q[mask, 1, 3]
    arr_transforms[mask, 2, 1] = q[mask, 1, 2] + q[mask, 0, 3]
    arr_transforms[mask, 2, 2] = 1.0 - q[mask, 0, 0] - q[mask, 1, 1]
    return arr_transforms


def ominus(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    """Computes the result of applying the inverse motion composition operator on two pose matrices. This implementation
    is vectorized: arr_a and arr_b should be numpy.ndarray with shape (N, 4, 4) containing N 4x4 transformation
    matrices. The output will also be N 4x4 matrices.

    Args:
        arr_a (np.ndarray): Nx4x4 array of N poses (homogeneous 4x4 matrix)
        arr_b (np.ndarray): Nx4x4 array of N poses (homogeneous 4x4 matrix)

    Returns:
        np.ndarray: Nx4x4 array of N resulting 4x4 relative transformation matrices between a and b
    """
    return np.matmul(np.linalg.inv(arr_b), arr_a)


def compute_distance(arr_transforms: np.ndarray) -> np.ndarray:
    """
    Compute the distance (magnitude) of the translational components of an array of N 4x4 homogeneous matrices.

    Args:
        arr_transforms (np.ndarray): Nx4x4 array of N 4x4 transformation matrices

    Returns:
        np.ndarray: 1D array of length N with distance values
    """
    return np.linalg.norm(arr_transforms[:, :3, 3], axis=1)


def compute_angle(arr_transforms: np.ndarray) -> np.ndarray:
    """
    Compute the rotation angle from the rotational components an array of N 4x4 homogeneous matrices.

    Args:
        arr_transforms (np.ndarray): Nx4x4 array of N 4x4 transformation matrices

    Returns:
        np.ndarray: 1D array of length N with angle values in radians
    """
    trace_r = np.trace(arr_transforms[:, :3, :3], axis1=1, axis2=2)
    raw = (trace_r - 1) / 2
    clipped = np.clip(raw, a_min=-1, a_max=1)
    return np.arccos(clipped)


def normalized_pose_errors(predicted: np.ndarray, actual: np.ndarray) -> Dict[str, numpy.typing.NDArray[float]]:
    """Calculate normalized rotation and translation pose errors for a set predictions against the ground truth.

    This is the absolute pose error. It assumes that poses are described in the reference frame of the initial image of
    the sequence (the "reference pose"). This means the reference pose should be the null pose, i.e., it has a rotation
    component that is the identity and zero translation component.

    The rotation and translation errors are both normalized by the magnitude of the ground truth rotation and
    translation magnitudes, respectively.

    The input array columns should correspond to the position and quaternion coordinates:
    ["x", "y", "z", "qw", "qx", "qy", "qz"]

    Args:
        predicted (np.ndarray): Nx7 array of predicted pose coordinates
        actual (np.ndarray): Nx7 array of ground truth pose coordinates

    Returns:
        dict[str, np.ndarray[float]]: dictionary of arrays of rotation and translation errors
    """
    m, n = actual.shape
    assert n == len(POSE_COLS)
    assert actual.shape == predicted.shape

    # Convert translation vector + rotation quaternions to 4x4 transformation matrices
    predicted_arr_transforms = transformation_matrix(predicted[:, :3], predicted[:, 3:])
    actual_arr_transforms = transformation_matrix(actual[:, :3], actual[:, 3:])

    # First frame is reference frame. We will calculate errors for all subsequent frames
    j_predicted = predicted_arr_transforms[1:, :, :]
    j_actual = actual_arr_transforms[1:, :, :]

    # Absolute pose error of 4x4 transformation matrices
    error44 = ominus(j_predicted, j_actual)

    # Translation error, normalized by magnitude of ground truth translation
    trans_err = compute_distance(error44)
    trans_denom = np.clip(compute_distance(j_actual), a_min=_EPS, a_max=np.inf)  # can't be zero
    corrected_trans_err = trans_err / trans_denom
    corrected_trans_err = np.clip(corrected_trans_err, a_min=0, a_max=TRANS_MAX_ERROR)
    corrected_trans_err = np.where(
        corrected_trans_err >= TRANS_ERR_THRESHOLD, corrected_trans_err, 0.0
    )

    # Rotation error, normalized by magnitude of ground truth rotation
    rot_err = compute_angle(error44)
    rot_denom = np.clip(
        compute_angle(j_actual), a_min=_EPS, a_max=np.inf
    )  # can't be zero
    corrected_rot_err = rot_err / rot_denom
    corrected_rot_err = np.where(
        corrected_rot_err >= ROT_ERR_THRESHOLD, corrected_rot_err, 0.0
    )

    return {
        "translation_errors": trans_err,
        "normalized_translation_errors": corrected_trans_err,
        "rotation_errors": rot_err,
        "normalized_rotation_errors": corrected_rot_err,
    }


def score_chains(predicted_df: pd.DataFrame, actual_df: pd.DataFrame) -> Dict[str, float]:
    """Scores chains of predicted object poses (trajectory of object) against the ground truth. Returns the
    micro-average of the normalized absolute translation and rotation errors over all sequences, and the pose error
    score which is the weighted sum of the two.

    Args:
        predicted_df (np.ndarray): dataframe of predicted pose coordinates
        actual_df (np.ndarray): dataframe of ground truth pose coordinates

    Returns:
        dict[str, float]: dictionary of errors and score
    """
    errors = {}
    chain_ids = actual_df.index.get_level_values(0).unique().values.tolist()
    for chain_id in chain_ids:
        predicted_i = predicted_df.loc[chain_id]
        """
        if not np.allclose(
            predicted_i.loc[0], REFERENCE_VALUES, atol=REFERENCE_TOLERANCE
        ):
            raise ValueError(
                f"Reference row for chain {chain_id} not close enough (|δ| ≤ {REFERENCE_TOLERANCE}) to expected values"
                f"\n    expected: {REFERENCE_VALUES}"
                f"\n    actual: {predicted_i.loc[0].values}"
            )
        """
        actual_i = actual_df.loc[chain_id]
        errors[chain_id] = normalized_pose_errors(predicted_i.values, actual_i.values)

    mean_translation_error = np.concatenate(
        [
            chain_errors["normalized_translation_errors"]
            for chain_errors in errors.values()
        ]
    ).mean()
    mean_rotation_error = np.concatenate(
        [chain_errors["normalized_rotation_errors"] for chain_errors in errors.values()]
    ).mean()
    
    error_dict = {}
    for k in errors:
        error_dict[k] = errors[k]["normalized_translation_errors"]+errors[k]["normalized_rotation_errors"] 

    return {
        "mean_translation_error": mean_translation_error,
        "mean_rotation_error": mean_rotation_error,
        "score": mean_translation_error + mean_rotation_error,
    }, error_dict


def get_pred_df(preds, chain_ids, seids, label_scaler=None):
    data0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    data0_df = pd.DataFrame({'chain_id':np.unique(chain_ids), 'i':0})
    data0_df[LABEL_NAMES] = data0
    
    predicted_df = pd.DataFrame({'chain_id':chain_ids, 'i':seids})
    predicted_df[LABEL_NAMES] = preds
    predicted_df = pd.concat([data0_df, predicted_df]).sort_values(['chain_id', 'i'])
    return predicted_df.reset_index(drop=True)

def get_score(preds, labels, chain_ids, seids):
    predicted_df = get_pred_df(preds, chain_ids, seids)
    actual_df = get_pred_df(labels, chain_ids, seids)
    #print(predicted_df)
    #print(actual_df)
    
    scores, error_dict = score_chains(predicted_df.set_index(INDEX_COLS).loc[:, POSE_COLS], 
                                  actual_df.set_index(INDEX_COLS).loc[:, POSE_COLS])
    print(scores)
    return scores['score'], error_dict, predicted_df, actual_df

# my loss
def transformation_matrix_torch(t, q):
    # Allocate transformation matrices with only translation
    arr_transforms = torch.tensor(np.repeat(np.eye(4)[None, :, :], t.shape[0], axis=0))
    arr_transforms = arr_transforms.to(t.device).to(t.dtype)
    arr_transforms[:, :3, 3] = t

    # shift so we have (qx, qy, qz, qw) instead of (qw, qx, qy, qz)
    q = torch.cat([q[...,1:], q[...,:1]], dim=-1)

    nq = torch.square(torch.linalg.norm(q, dim=1))
    mask = nq >= _EPS  # mask for rotations of magnitude greater than epsilon

    # For transformations with non-zero rotation, calculate rotation matrix
    q = torch.sqrt(2.0 / nq)[:, None] * q
    q = q[:, :, None] * q[:, None, :]  # outer product
    arr_transforms[mask, 0, 0] = 1.0 - q[mask, 1, 1] - q[mask, 2, 2]
    arr_transforms[mask, 0, 1] = q[mask, 0, 1] - q[mask, 2, 3]
    arr_transforms[mask, 0, 2] = q[mask, 0, 2] + q[mask, 1, 3]
    arr_transforms[mask, 1, 0] = q[mask, 0, 1] + q[mask, 2, 3]
    arr_transforms[mask, 1, 1] = 1.0 - q[mask, 0, 0] - q[mask, 2, 2]
    arr_transforms[mask, 1, 2] = q[mask, 1, 2] - q[mask, 0, 3]
    arr_transforms[mask, 2, 0] = q[mask, 0, 2] - q[mask, 1, 3]
    arr_transforms[mask, 2, 1] = q[mask, 1, 2] + q[mask, 0, 3]
    arr_transforms[mask, 2, 2] = 1.0 - q[mask, 0, 0] - q[mask, 1, 1]
    return arr_transforms

def ominus_torch(arr_a, arr_b):
    return torch.matmul(torch.linalg.inv(arr_b), arr_a)

def compute_distance_torch(arr_transforms):
    return torch.linalg.norm(arr_transforms[:, :3, 3], axis=1)

def compute_angle_torch(arr_transforms):
    trace_r = arr_transforms[:, :3, :3].diagonal(offset=0, dim1=1, dim2=2).sum(-1)
    raw = (trace_r - 1) / 2
    clipped = torch.clamp(raw, min=-1, max=1)
    return torch.arccos(clipped)

def normalized_pose_errors_torch(predicted, actual):
    m, n = actual.shape
    #print(actual.shape, predicted.shape)
    assert n == len(POSE_COLS)
    assert actual.shape == predicted.shape

    # Convert translation vector + rotation quaternions to 4x4 transformation matrices
    predicted_arr_transforms = transformation_matrix_torch(predicted[:, :3], predicted[:, 3:])
    actual_arr_transforms = transformation_matrix_torch(actual[:, :3], actual[:, 3:])

    # First frame is reference frame. We will calculate errors for all subsequent frames
    j_predicted = predicted_arr_transforms#[1:, :, :]
    j_actual = actual_arr_transforms#[1:, :, :]

    # Absolute pose error of 4x4 transformation matrices
    error44 = ominus_torch(j_predicted, j_actual)

    # Translation error, normalized by magnitude of ground truth translation
    trans_err = compute_distance_torch(error44)
    trans_denom = torch.clamp(compute_distance_torch(j_actual), min=_EPS, max=np.inf)  # can't be zero
    corrected_trans_err = trans_err / trans_denom
    corrected_trans_err = torch.clamp(corrected_trans_err, min=0, max=TRANS_MAX_ERROR)

    # Rotation error, normalized by magnitude of ground truth rotation
    rot_err = compute_angle_torch(error44)
    rot_denom = torch.clamp(
        compute_angle_torch(j_actual), min=_EPS, max=np.inf
    )  # can't be zero
    corrected_rot_err = rot_err / rot_denom
    """
    preds2 = predicted[:, 3:]
    labels2 = actual[:, 3:]
    corrected_rot_err = min([base_loss(preds2, labels2), base_loss(preds2, -labels2)])
    """
    return corrected_trans_err + corrected_rot_err

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to rotation matrix.
    """
    w, x, y, z = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]

    rotation_matrix = torch.stack([
        torch.stack([1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)], dim=1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)], dim=1)
    ], dim=1)

    return rotation_matrix

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert rotation matrix to quaternion.
    """
    trace = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]

    mask = trace > 0

    s = torch.sqrt(trace[mask] + 1.0)
    q = torch.zeros(rotation_matrix.size(0), 4, device=rotation_matrix.device)
    q[mask, 0] = 0.5 * s
    q[mask, 1] = 0.25 / s * (rotation_matrix[mask, 2, 1] - rotation_matrix[mask, 1, 2])
    q[mask, 2] = 0.25 / s * (rotation_matrix[mask, 0, 2] - rotation_matrix[mask, 2, 0])
    q[mask, 3] = 0.25 / s * (rotation_matrix[mask, 1, 0] - rotation_matrix[mask, 0, 1])

    indices = torch.argmax(trace, dim=0)
    for idx in range(rotation_matrix.size(0)):
        if idx not in mask:
            print(mask.shape, indices, idx)
            i = indices[idx]
            if i == 0:
                s = 2.0 * torch.sqrt(1.0 + rotation_matrix[idx, 0, 0] - rotation_matrix[idx, 1, 1] - rotation_matrix[idx, 2, 2])
                q[idx, 1] = (rotation_matrix[idx, 0, 1] + rotation_matrix[idx, 1, 0]) / s
                q[idx, 2] = 0.25 * s
                q[idx, 3] = (rotation_matrix[idx, 0, 2] + rotation_matrix[idx, 2, 0]) / s
                q[idx, 0] = (rotation_matrix[idx, 1, 2] - rotation_matrix[idx, 2, 1]) / s
            elif i == 1:
                s = 2.0 * torch.sqrt(1.0 - rotation_matrix[idx, 0, 0] + rotation_matrix[idx, 1, 1] - rotation_matrix[idx, 2, 2])
                q[idx, 1] = (rotation_matrix[idx, 0, 1] + rotation_matrix[idx, 1, 0]) / s
                q[idx, 2] = (rotation_matrix[idx, 1, 2] + rotation_matrix[idx, 2, 1]) / s
                q[idx, 3] = 0.25 * s
                q[idx, 0] = (rotation_matrix[idx, 0, 2] - rotation_matrix[idx, 2, 0]) / s
            else: # i == 2
                s = 2.0 * torch.sqrt(1.0 - rotation_matrix[idx, 0, 0] - rotation_matrix[idx, 1, 1] + rotation_matrix[idx, 2, 2])
                q[idx, 1] = (rotation_matrix[idx, 0, 2] + rotation_matrix[idx, 2, 0]) / s
                q[idx, 2] = (rotation_matrix[idx, 1, 2] + rotation_matrix[idx, 2, 1]) / s
                q[idx, 3] = (rotation_matrix[idx, 2, 0] - rotation_matrix[idx, 0, 2]) / s
                q[idx, 0] = 0.25 * s

    return q

def calculate_angle_differences(angles):
    """
    Calculate angle differences between consecutive quaternion angles.
    
    Args:
    - angles: Tensor of shape (seq_len, 4) containing a sequence of quaternion angles
    
    Returns:
    - angle_diffs: Tensor of shape (seq_len - 1) containing the angle differences between consecutive angles
    """
    # Convert quaternions to rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(angles)
    
    # Compute relative rotations between consecutive frames
    rotation_matrices[1:] = torch.matmul(rotation_matrices[1:], torch.inverse(rotation_matrices[:-1]))
    relative_rotations = rotation_matrices

    # Convert relative rotations back to quaternions
    relative_quaternions = rotation_matrix_to_quaternion(relative_rotations)
    
    # Calculate angle differences between consecutive quaternions
    angle_diffs = 2 * torch.acos(torch.abs(relative_quaternions[:, 0]))
    
    # Convert angle differences to degrees
    angle_diffs_deg = angle_diffs * 180 / torch.pi
    
    return angle_diffs_deg

def get_loss2(preds, all_ranges):
    part1 = 0
    xyzs = preds[:,:3]
    xyzs[:, 0] = all_ranges[0] - xyzs[:, 0]
    _part = torch.sum((xyzs[1:]-xyzs[:-1])**2, dim=1) ** 0.5
    _part = _part[_part>1]
    _count = 0
    if len(_part) > 0:
        part1 += torch.sum(_part)
        _count += len(_part)
    _part = sum([(xyzs[0][0]+all_ranges[0])**2, xyzs[0][1]**2, xyzs[0][2]**2]) ** 0.5
    if _part > 1: 
        part1 += _part
        _count += 1
    if _count > 0:
        part1 /= _count
        
    _part = torch.sum(xyzs**2, dim=1) ** 0.5
    part2 = torch.mean((all_ranges[1:]-_part)**2) 
    #print(part1, part2)
    #angle_pad = torch.tensor([[1, 0, 0, 0]], dtype=preds.dtype).to(preds.device)
    #all_angles = torch.cat([angle_pad, preds[:, 3:]], dim=0)
    #delta_angles = calculate_angle_differences(all_angles)
    #part3 = torch.sum(delta_angles[delta_angles>30]*10) 
    
    return part1 + part2 #+ part3

def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.

    Args:
    - quaternions: Tensor of shape (batch_size, 4) containing quaternion rotations

    Returns:
    - rotation_matrices: Tensor of shape (batch_size, 3, 3) containing rotation matrices
    """
    # Extract quaternion components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Calculate rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # Calculate rotation matrix
    rotation_matrices = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=1).view(-1, 3, 3)
    
    return rotation_matrices

def rotate_vector_by_quaternion(vectors, quaternions):
    """
    Rotate vectors by quaternions.

    Args:
    - vectors: Tensor of shape (batch_size, 3) containing 3D vectors to be rotated
    - quaternions: Tensor of shape (batch_size, 4) containing quaternion rotations

    Returns:
    - rotated_vectors: Tensor of shape (batch_size, 3) containing rotated vectors
    """
    # Ensure inputs are float tensors
    vectors = vectors
    quaternions = quaternions

    # Extract quaternion components
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    # Calculate intermediate terms
    wx = w * x
    wy = w * y
    wz = w * z
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = y * z
    zz = z * z

    # Calculate rotation matrix elements
    rotation_matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=1).view(-1, 3, 3)

    # Rotate vectors using rotation matrices
    rotated_vectors = torch.bmm(rotation_matrix, vectors.unsqueeze(-1)).squeeze(-1)

    return rotated_vectors

range_penalty_rate = 0.1
def loss2_func(preds, all_ranges, max_rotation_rad=30, max_movement=1):
    predictions = preds * 1
    predictions[:, 0] = all_ranges[0] - predictions[:, 0]
    
    # Extract camera poses from predictions
    predicted_locations = predictions[:, :3]
    predicted_quaternions = predictions[:, 3:]

    # Calculate rotation angles
    #rotation_angles = calculate_angle_differences(predicted_quaternions)

    # Calculate movement distances between sequential outputs
    #movement_distances = torch.norm(predicted_locations, dim=1)
    #movement_distances[1:] = movement_distances[1:] - movement_distances[:-1]

    # Calculate direction vector from camera to origin
    direction_to_origin = -predicted_locations / torch.norm(predicted_locations, dim=1, keepdim=True)

    # Calculate forward direction of the camera (Z-axis in camera space)
    forward_direction = torch.tensor([[-1, 0, 0]], dtype=torch.float32, 
                                     device=predicted_locations.device).expand_as(predicted_locations)
    forward_direction = rotate_vector_by_quaternion(forward_direction, predicted_quaternions)
    forward_direction[:, 1:] = -forward_direction[:, 1:]
    #print(direction_to_origin, forward_direction)
    # Calculate dot product between direction to origin and forward direction
    dot_products = torch.sum(direction_to_origin * forward_direction, dim=1)
    #dot_products = dot_products[~torch.isnan(dot_products)]

    # Penalize deviation from the camera always facing the origin
    alignment_penalty = torch.mean(torch.abs(dot_products - 1))

    # Penalize rotation angles exceeding the threshold
    #rotation_penalty = torch.sum(torch.where(rotation_angles > max_rotation_rad,
    #                                          rotation_angles - max_rotation_rad,
    #                                          torch.tensor(0.0)))

    # Penalize movement distances exceeding the threshold
    #movement_penalty = torch.sum(torch.where(movement_distances > max_movement,
    #                                         movement_distances - max_movement,
    #                                         torch.tensor(0.0)))
    
    # Calculate distance from camera to origin
    predicted_ranges = torch.norm(predicted_locations, dim=1)
    #predicted_ranges = predicted_ranges[~torch.isnan(predicted_ranges)]
    range_mask = ((all_ranges[1:]-all_ranges[:1])**2) ** 0.5 <= 1
    range_penalty = ((predicted_ranges-all_ranges[1:])**2) ** 0.5
    range_penalty = range_penalty - all_ranges[1:]*0.2
    range_penalty = torch.mean(torch.where((range_penalty>0)&range_mask,
                                  range_penalty,
                                  torch.tensor(0.0)))

    # Combine penalties into total loss
    return alignment_penalty, range_penalty*range_penalty_rate

def my_loss(preds, labels, all_ranges):
    
    loss = normalized_pose_errors_torch(preds, labels)
    #loss[loss>2] *= 2
    #loss = loss[~torch.isnan(loss)]
    loss = loss.mean()
    """
    loc_loss = torch.mean(torch.norm(preds[:, :3]-labels[:, :3], dim=-1))
    rot_loss1 = torch.mean(torch.norm(preds[:, :3]-labels[:, :3], dim=-1))
    rot_loss2 = torch.mean(torch.norm(preds[:, 3:]+labels[:, 3:], dim=-1))
    rot_loss = min([rot_loss1, rot_loss2])
    loss = loc_loss + rot_loss
    """
    
    loss2 = sum(loss2_func(preds, all_ranges))
    return loss + loss2

class CFG:
    seed = 1086
    lag = 0
    nrots = 4
    used_nrots = 4
    n_feats = 2
    newfeat_lags = [1, 2, 3]
N_ROTS = 4
LAG = 0
    
LABEL_NAMES = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
RANDAM_SEED = CFG.seed


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.

    Args:
    - quaternions: Tensor of shape (batch_size, 4) containing quaternion rotations

    Returns:
    - rotation_matrices: Tensor of shape (batch_size, 3, 3) containing rotation matrices
    """
    # Extract quaternion components
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    
    # Calculate rotation matrix elements
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    # Calculate rotation matrix
    rotation_matrices = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=1).view(-1, 3, 3)
    
    return rotation_matrices

def rotate_vector_by_quaternion(vectors, quaternions):
    """
    Rotate vectors by quaternions.

    Args:
    - vectors: Tensor of shape (batch_size, 3) containing 3D vectors to be rotated
    - quaternions: Tensor of shape (batch_size, 4) containing quaternion rotations

    Returns:
    - rotated_vectors: Tensor of shape (batch_size, 3) containing rotated vectors
    """
    # Ensure inputs are float tensors
    vectors = vectors
    quaternions = quaternions

    # Extract quaternion components
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]

    # Calculate intermediate terms
    wx = w * x
    wy = w * y
    wz = w * z
    xx = x * x
    xy = x * y
    xz = x * z
    yy = y * y
    yz = y * z
    zz = z * z

    # Calculate rotation matrix elements
    rotation_matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=1).view(-1, 3, 3)

    # Rotate vectors using rotation matrices
    rotated_vectors = torch.bmm(rotation_matrix, vectors.unsqueeze(-1)).squeeze(-1)

    return rotated_vectors

range_penalty_rate = 0.1
def loss2_func(preds, all_ranges, max_rotation_rad=30, max_movement=1):
    predictions = preds * 1
    predictions[:, 0] = all_ranges[0] - predictions[:, 0]
    
    # Extract camera poses from predictions
    predicted_locations = predictions[:, :3]
    predicted_quaternions = predictions[:, 3:]

    # Calculate rotation angles
    #rotation_angles = calculate_angle_differences(predicted_quaternions)

    # Calculate movement distances between sequential outputs
    #movement_distances = torch.norm(predicted_locations, dim=1)
    #movement_distances[1:] = movement_distances[1:] - movement_distances[:-1]

    # Calculate direction vector from camera to origin
    direction_to_origin = -predicted_locations / torch.norm(predicted_locations, dim=1, keepdim=True)

    # Calculate forward direction of the camera (Z-axis in camera space)
    forward_direction = torch.tensor([[-1, 0, 0]], dtype=torch.float32, 
                                     device=predicted_locations.device).expand_as(predicted_locations)
    forward_direction = rotate_vector_by_quaternion(forward_direction, predicted_quaternions)
    forward_direction[:, 1:] = -forward_direction[:, 1:]
    #print(direction_to_origin, forward_direction)
    # Calculate dot product between direction to origin and forward direction
    dot_products = torch.sum(direction_to_origin * forward_direction, dim=1)
    #dot_products = dot_products[~torch.isnan(dot_products)]

    # Penalize deviation from the camera always facing the origin
    alignment_penalty = torch.mean(torch.abs(dot_products - 1))

    # Penalize rotation angles exceeding the threshold
    #rotation_penalty = torch.sum(torch.where(rotation_angles > max_rotation_rad,
    #                                          rotation_angles - max_rotation_rad,
    #                                          torch.tensor(0.0)))

    # Penalize movement distances exceeding the threshold
    #movement_penalty = torch.sum(torch.where(movement_distances > max_movement,
    #                                         movement_distances - max_movement,
    #                                         torch.tensor(0.0)))
    
    # Calculate distance from camera to origin
    predicted_ranges = torch.norm(predicted_locations, dim=1)
    #predicted_ranges = predicted_ranges[~torch.isnan(predicted_ranges)]
    #if len(predicted_ranges) > 0:
    range_mask = ((all_ranges[1:]-all_ranges[:1])**2) ** 0.5 <= 1
    range_penalty = ((predicted_ranges-all_ranges[1:])**2) ** 0.5
    range_penalty = torch.mean(torch.where((range_penalty>all_ranges[1:]*0.15)&range_mask,
                                  range_penalty,
                                  torch.tensor(0.0)))

    # Combine penalties into total loss
    return alignment_penalty, range_penalty*range_penalty_rate

def rotation_matrix_to_quaternion(rotation):
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    q = np.zeros(4)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        q[0] = 0.25 / s
        q[1] = (rotation[2][1] - rotation[1][2]) * s
        q[2] = (rotation[0][2] - rotation[2][0]) * s
        q[3] = (rotation[1][0] - rotation[0][1]) * s
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        s = 2.0 * np.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2])
        q[0] = (rotation[2][1] - rotation[1][2]) / s
        q[1] = 0.25 * s
        q[2] = (rotation[0][1] + rotation[1][0]) / s
        q[3] = (rotation[0][2] + rotation[2][0]) / s
    elif rotation[1][1] > rotation[2][2]:
        s = 2.0 * np.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2])
        q[0] = (rotation[0][2] - rotation[2][0]) / s
        q[1] = (rotation[0][1] + rotation[1][0]) / s
        q[2] = 0.25 * s
        q[3] = (rotation[1][2] + rotation[2][1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1])
        q[0] = (rotation[1][0] - rotation[0][1]) / s
        q[1] = (rotation[0][2] + rotation[2][0]) / s
        q[2] = (rotation[1][2] + rotation[2][1]) / s
        q[3] = 0.25 * s

    return q / np.linalg.norm(q)

FEATS= [
    #cv2.SIFT_create(),
    cv2.ORB_create(scaleFactor=1.1, nlevels=12, edgeThreshold=10, patchSize=21),
    cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, 
                     threshold=0.0005, nOctaves=8),
    #cv2.xfeatures2d.SURF_create(),
    #cv2.BRISK_create(),
    #cv2.FastFeatureDetector_create(),
    #cv2.GFTTDetector_create(),
    #cv2.SimpleBlobDetector_create()
]
# Create BFMatcher object
BFM = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
def get_rotations(keypoints_list, descriptors_list, rots, 
                  new_image, valid_count, all_count, intrinsics, lag):
    
    # Detect keypoints and compute descriptors
    for fi, feat in enumerate(FEATS):
        keypoints, descriptors = feat.detectAndCompute(new_image, None)
        keypoints_list[fi].append(keypoints)
        descriptors_list[fi].append(descriptors)
        
    if len(keypoints_list[0]) == 1:
        new_rot = np.array([[1.0, 0.0, 0.0, 0.0]*(lag+1)*N_ROTS]*len(FEATS))
    else:
        last_rot = rots[-1]
        quaternion_list = []
        for fi, feat in enumerate(FEATS):
            i = len(keypoints_list[0])-1
            _quaternion = []
            for li in range(lag+1):
                i0 = i - li - 1
                if i0 < 0:
                    _quaternion.append(_quaternion[-1])
                else:
                    #print(f'{fi}, {i0}, {len(keypoints_list)}, {len(keypoints_list[0])}')
                    keypoints1, keypoints2 = keypoints_list[fi][i0], keypoints_list[fi][i]
                    descriptors1, descriptors2 = descriptors_list[fi][i0], descriptors_list[fi][i]
                    try:
                        # Match keypoints
                        matches = BFM.match(descriptors1, descriptors2)

                        """
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                        """
                        # Sort matches by distance
                        matches = sorted(matches, key=lambda x: x.distance)

                        # Select top matches
                        good_matches = matches[:200]

                        # Extract corresponding keypoints
                        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        retval, rotations, translation, normals = cv2.decomposeHomographyMat(H, intrinsics)

                        rotations = [rotation_matrix_to_quaternion(rot) for rot in rotations]
                        rotations = sorted(rotations, key=lambda x:-abs(x[0]))
                        if len(rotations) < N_ROTS:
                            rotations += [rotations[-1]] * (N_ROTS-len(rotations))
                        else:
                            rotations = rotations[:N_ROTS]
                        _quaternion.append(np.array(rotations).reshape(-1))
                        #_nan_ids.append(0)
                        valid_count += 1
                    except:
                        _quaternion.append([np.nan, np.nan, np.nan, np.nan]*N_ROTS)
                        #_nan_ids.append(1)
                    all_count += 1
                quaternion_list.append(np.array(_quaternion).reshape(-1))
                #nan_ids.append(_nan_ids)
        new_rot = np.array(quaternion_list)
        #print(new_rot, new_rot.shape)
        #nan_ids = np.isnan(new_rot)
        #assert nan_ids.sum() == 0
        #print(f'{new_rot.shape}, {last_rot.shape}')
        #new_rot[nan_ids] = last_rot[nan_ids]
    return keypoints_list, descriptors_list, new_rot, valid_count, all_count

def generate_one(xs, rs, r0, models, past_key_values_list):
    y_loc = []
    y_rot = []
    new_past_key_values_list = []
    with torch.no_grad():
        for model, past_key_values in zip(models, past_key_values_list):
            y, past_key_values = model.predict(xs, rs, r0, past_key_values)
            y = y.detach().cpu().numpy()[0]
            y_loc.append(y[:3])
            y_rot.append(y[3:])
            new_past_key_values_list.append(past_key_values)
    y_loc = np.mean(y_loc, axis=0)
    y_rot = quaternion_average(y_rot)
    y = np.concatenate([y_loc, y_rot], axis=-1) 
    
    return y, new_past_key_values_list

class PE_Dataset(torch.utils.data.Dataset):
    def __init__(self, df_file, data_folders, models):
        self.data_folders = data_folders
        self.models = models
        self.lag = LAG
        self.intrinsics = np.array([[5.2125371e+03, 0.0000000e+00, 6.4000000e+02],
            [0.0000000e+00, 6.2550444e+03, 5.1200000e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
       
        self.data_paths = {}
        self.ranges = {}
        self.seq_ids = {}
        self.chain_ids = []
        for ci, label in df_file.groupby('chain_id'):
            for folder in data_folders:
                paths = glob(f'{folder}{ci}/*.*')
                if len(paths) > 0:
                    break
            assert len(paths) > 0
            self.data_paths[ci] = sorted(paths)
            self.seq_ids[ci] = [int(path.split('/')[-1][:-4]) for path in self.data_paths[ci]]
            label = label.sort_values('i')
            self.ranges[ci] = label['range'].values
            self.chain_ids.append(ci)
        
    def __len__(self):
        return len(self.chain_ids)
    
    def __getitem__(self, idc):
        ci = self.chain_ids[idc]
        used_paths = self.data_paths[ci]
        used_ranges = self.ranges[ci]
        used_seids = self.seq_ids[ci]
        
        if np.isnan(used_ranges[0]):
            used_ranges[0] = 264.7
            #print(f'{ci} frist range is nan')
        
        # ------------------------------ predict one by one --------------------------------------
        keypoints_list = [[] for _ in range(len(FEATS))]
        descriptors_list = [[] for _ in range(len(FEATS))]
        rots = []
        valid_count = 0
        all_count = 0
        past_key_values_list = [None for _ in range(len(self.models))]
        out_list = []
        r0 = torch.tensor(used_ranges[0], dtype=torch.float32)
        
        for i, (p, r) in enumerate(zip(used_paths, used_ranges)):
            image = cv2.imread(p)
            keypoints_list, descriptors_list, rot, valid_count, all_count = get_rotations(keypoints_list, 
                                                                                        descriptors_list, rots, 
                                                                                        image, valid_count,
                                                                                        all_count, self.intrinsics,
                                                                                        self.lag)
            if i > 0:
                _nan_ids = np.isnan(rot)
                rot[_nan_ids] = rots[-1][_nan_ids]
            #assert np.isnan(rots[i0]).sum() == 0
            rots.append(rot)
            new_rot = []
            for i1 in range(rot.shape[0]):
                new_rot.append([])
                for i2 in range(0, rot.shape[1], 4):
                    _rot_cache = list(rot[i1, i2:i2+4])
                    for newfeat_lag in CFG.newfeat_lags:
                        if len(rots) <= newfeat_lag+1:
                            _rot_cache += list(rot[i1, i2:i2+4])
                        else:
                            _rot_cache += list(apply_delta_quaternion(rot[i1, i2:i2+4],
                                                                     rots[-newfeat_lag-1][i1, :4]))
                    new_rot[-1].append(_rot_cache)
            xs = torch.tensor(np.array(new_rot).reshape(-1), dtype=torch.float32)
            
            if np.isnan(r):
                r = used_ranges[i-1]
                used_ranges[i] = r
            rs = r / 300
            rs = torch.tensor(rs, dtype=torch.float32)
            #print(rs)
            pred, past_key_values_list = generate_one(xs, rs, r0, self.models, past_key_values_list)
            output_nans = np.isnan(pred).sum()
            #if output_nans > 0:
            #    print(f'output_nans: {output_nans}')
            out_list.append(pred)
            
        #print(f'{ci} nan rate: {min([len(nanids[0][]) for i in nanids[0]])/len(rots[0])}')
        valid_rate = valid_count / all_count
        
        return np.array(out_list[1:]), ci, used_seids[1:], valid_rate
    
class PE_Model(nn.Module):
    def predict(self, rot, rs, r0, past_key_values=None):
        # img: the single 224*224 image input
        # rg: the range value of the input image
        # past_key_values: the model's memory
        embed = torch.cat([self.trans(rot[None]), rs[..., None, None]], dim=-1)
        decoder_out = self.decoder(inputs_embeds=embed[None], use_cache=True,
                                   past_key_values=past_key_values)
        out = decoder_out.last_hidden_state[0]
        past_key_values = decoder_out.past_key_values
        out = self.head(out)
        loc_offset = self.head2(out)
        out2 = out[:, -4:] / torch.linalg.norm(out[:, -4:], dim=-1, keepdim=True)
        #out = self.head2(out) * 400
        base_loc = torch.tensor([[-r0, 0, 0]]*out2.shape[0], dtype=out2.dtype, 
                                     device=out2.device)
        out = rotate_vector_by_quaternion(base_loc, out2) - base_loc
        out = torch.cat([out+loc_offset, out2], dim=1)
        return out, past_key_values
    
def quaternion_average(quaternions):
    """
    Compute the average quaternion from a list of quaternions using Slerp.
    
    Parameters:
        quaternions (list of numpy arrays): List of quaternions (w, x, y, z).
        
    Returns:
        numpy array: Average quaternion (w, x, y, z).
    """
    # Convert quaternions to unit quaternions
    quaternions = np.array(quaternions)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)

    # Compute the average quaternion using Slerp
    avg_quaternion = np.zeros(4)
    for q in quaternions:
        if np.dot(avg_quaternion, q) < 0:
            q = -q  # Ensure shortest path
        avg_quaternion += q
    avg_quaternion /= np.linalg.norm(avg_quaternion)

    return avg_quaternion

def quaternions_average(quaternions):
    quaternions = np.array(quaternions)
    out = []
    for i in range(quaternions.shape[1]):
        out.append(quaternion_average(quaternions[:, i]))
    return np.array(out)

def get_pred_df(preds, chain_ids, seids, label_scaler=None):
    data0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    
    data0_df = pd.DataFrame({'chain_id':np.unique(chain_ids), 'i':0})
    data0_df[LABEL_NAMES] = data0
    
    predicted_df = pd.DataFrame({'chain_id':chain_ids, 'i':seids})

    predicted_df[LABEL_NAMES] = preds
    predicted_df = pd.concat([data0_df, predicted_df]).sort_values(['chain_id', 'i'])
    #predicted_df[['qw', 'qx', 'qy', 'qz']] = [1, 0, 0, 0]
    return predicted_df.reset_index(drop=True)