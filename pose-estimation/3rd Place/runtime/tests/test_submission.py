from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_index_equal

SUBMISSION_PATH = Path("/code_execution/submission/submission.csv")
SUBMISSION_FORMAT_PATH = Path("/code_execution/data/submission_format.csv")
INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]
REFERENCE_VALUES = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
REFERENCE_TOLERANCE = 1e-6


def test_submission_exists():
    assert SUBMISSION_PATH.exists(), f"Expected submission at {SUBMISSION_PATH}"


def test_submission_matches_submission_format():
    assert SUBMISSION_FORMAT_PATH.exists(), "Submission format not found"

    submission = pd.read_csv(SUBMISSION_PATH)
    fmt = pd.read_csv(SUBMISSION_FORMAT_PATH)

    assert_array_equal(submission.columns, fmt.columns, err_msg="Columns not identical")
    assert_index_equal(submission.index, fmt.index), "Index not identical"

    for col in submission.columns:
        assert submission[col].dtype == fmt[col].dtype, f"dtype for column {col} is not {fmt[col].dtype}"
        assert submission[col].notnull().all(), f"Missing values found in submission column {col}"

    for col in PREDICTION_COLS:
        assert np.isfinite(submission[col]).all(), f"Non-finite values found in submission column {col}"


def test_submission_reference_rows_within_atol():
    submission = pd.read_csv(SUBMISSION_PATH)
    reference_mask = submission.i == 0
    reference_rows = (
        submission.loc[reference_mask].set_index("chain_id").drop(columns=["i"])
    )
    for chain_id in reference_rows.index.values.tolist():
        err_msg = f"Reference row for chain {chain_id} too far from expected values (|δ| > {REFERENCE_TOLERANCE})"
        values_i = reference_rows.loc[chain_id].values.ravel()
        assert_allclose(
            values_i, REFERENCE_VALUES, atol=REFERENCE_TOLERANCE, err_msg=err_msg
        )
