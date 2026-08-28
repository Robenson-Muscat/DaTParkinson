from pathlib import Path

from loguru import logger
import pandas as pd

from model import DATScanModel


DATA_DIR = Path("/code_execution/data")

NIFTI_DIR = DATA_DIR / "niftis"

SUBMISSION_FORMAT_PATH = (DATA_DIR / "submission_format.csv")

WRITE_SUBMISSION_PATH = Path("submission.csv")

SRC_ROOT = Path(__file__).parent.resolve()


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINTS = [
    Path("ckpts/best_DATScanModel.pth"),

]


def main() -> None:

    # ========================================================
    # Load submission format
    # ========================================================

    submission_format = pd.read_csv(
        SUBMISSION_FORMAT_PATH
    )

    logger.info(
        "Loaded submission_format.csv "
        f"with {len(submission_format)} rows."
    )

    logger.info(
        f"Submission columns: "
        f"{list(submission_format.columns)}"
    )

    # ========================================================
    # Identify UID column
    # ========================================================

    if "uid" not in submission_format.columns:
        raise ValueError(
            "submission_format.csv must contain "
            "a 'uid' column."
        )

    # ========================================================
    # Check checkpoints
    # ========================================================

    for checkpoint in CHECKPOINTS:

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

    logger.info(
        f"Using {len(CHECKPOINTS)} checkpoint(s)."
    )

    # ========================================================
    # Load model
    # ========================================================

    model = DATScanModel(
        checkpoint_paths=CHECKPOINTS,
    )

    logger.info(
        "Model loaded successfully."
    )

    # ========================================================
    # Predict
    # ========================================================

    predictions = []

    for _, row in submission_format.iterrows():

        uid = str(row["uid"])

        nifti_path = (
            NIFTI_DIR / f"{uid}.nii.gz"
        )

        if not nifti_path.exists():

            raise FileNotFoundError(
                f"NIfTI file not found for UID "
                f"{uid}: {nifti_path}"
            )

        probability = model.predict_one(
            nifti_path
        )

        predictions.append(
            probability
        )

        logger.info(
            f"{uid}: {probability:.6f}"
        )

    # ========================================================
    # Create submission
    # ========================================================

    submission = submission_format.copy()

    # Usually the submission target is
    # 'is_pathologic'.
    #
    # We detect it automatically if possible.

    if "is_pathologic" in submission.columns:

        target_column = "is_pathologic"

    else:

        non_uid_columns = [
            column
            for column in submission.columns
            if column != "uid"
        ]

        if len(non_uid_columns) != 1:

            raise ValueError(
                "Could not identify the prediction "
                "column automatically. "
                f"Columns are: "
                f"{list(submission.columns)}"
            )

        target_column = non_uid_columns[0]

    logger.info(
        f"Prediction column: {target_column}"
    )

    submission[target_column] = predictions

    # ========================================================
    # Save
    # ========================================================

    submission.to_csv(
        WRITE_SUBMISSION_PATH,
        index=False,
    )

    logger.info(
        f"Submission saved to: "
        f"{WRITE_SUBMISSION_PATH.resolve()}"
    )

    logger.info(
        f"Submission shape: "
        f"{submission.shape}"
    )

    logger.info(
        f"Prediction range: "
        f"[{min(predictions):.6f}, "
        f"{max(predictions):.6f}]"
    )


if __name__ == "__main__":
    main()