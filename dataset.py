import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

def create_cases_dataframe(df, nifti_dir):

    cases = []

    for _, row in df.iterrows():

        uid = row["uid"]

        cases.append({
            "image": str(nifti_dir / f"{uid}.nii.gz"),
            "label": float(row["is_pathologic"]),
            "uid": uid
        })

    return cases


def create_train_val_indices(cases):

    labels = np.array([c["label"] for c in cases])

    indices = np.arange(len(cases))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED
    )

    train_idx, val_idx = next(splitter.split(indices, labels))

    train_cases = [cases[i] for i in train_idx]

    val_cases = [cases[i] for i in val_idx]

    return train_cases, val_cases



