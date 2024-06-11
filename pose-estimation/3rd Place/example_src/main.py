from localizator import Localizator
from loguru import logger
import pandas as pd

INDEX_COLS = ["chain_id", "i"]
PREDICTION_COLS = ["x", "y", "z", "qw", "qx", "qy", "qz"]

def main():
    submission_format_path = "/code_execution/data/submission_format.csv"
    submission_format_df = pd.read_csv(
        submission_format_path, index_col=INDEX_COLS)
    ranges_df = pd.read_csv(
        "/code_execution/data/range.csv", index_col="chain_id")
    
    localizator = Localizator("/code_execution/data/")

    # copy over the submission format so we can overwrite placeholders with predictions
    submission_df = submission_format_df.copy()
    chain_ids = submission_format_df.index.get_level_values(0).unique()
    for chain_ord, chain_id in enumerate(chain_ids):
        logger.info(
            f"Processing chain: {chain_id}, {chain_ord + 1}/{len(chain_ids)}")
        chain_ranges = ranges_df[
            ranges_df.index.get_level_values(0) == chain_id]['range'].values
        chain_df = localizator.predict_chain(chain_id, chain_ranges)
        submission_df.loc[chain_id] = chain_df

    submission_df.to_csv(
        '/code_execution/submission/submission.csv', index=True)


if __name__ == "__main__":
    main()
