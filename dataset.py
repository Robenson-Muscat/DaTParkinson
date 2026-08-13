import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def create_cases_dataframe(df, nifti_dir):

    cases=[]

    for _, row in df.iterrows():

        uid = row["uid"]

        cases.append(
            {
                "image":
                str(
                    nifti_dir /
                    f"{uid}.nii.gz"
                ),

                "label":
                float(
                    row["is_pathologic"]
                ),

                "uid":
                uid
            }
        )

    return cases

def create_train_val_indices(cases):
    
    
    labels = np.array(
        [
            c["label"]
            for c in cases
        ]
    )
    
    
    indices = np.arange(
        len(cases)
    )
    
    
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED
    )
    
    
    train_idx, val_idx = next(
        splitter.split(
            indices,
            labels
        )
    )
    
    train_cases = [
        cases[i]
        for i in train_idx
    ]
    
    
    val_cases = [
        cases[i]
        for i in val_idx
    ]
    return train_cases, val_cases





class DaTScanDataset(Dataset):
    """
    Dataset PyTorch for  DaTScan volumes (.nii.gz). Useful for data analysis

    Parameters
    ----------
    data_dir : str | Path
        Folder with Files .nii.gz.
    csv_file : str | Path, optional
        CSV with labels.
    transform : callable, optional
        Preprocessing/ data augmentation.
    return_spacing : bool
        Returns the voxel spacing.
    """

    def __init__(
        self,
        data_dir,
        csv_file=None,
        transform=None,
        return_spacing=False,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_spacing = return_spacing

        if csv_file is not None:
            df = pd.read_csv(csv_file)

            self.uids = df["uid"].tolist()
            self.labels = df["is_pathologic"].astype(np.float32).tolist()

        else:
            self.uids = sorted(
                [f.stem.replace(".nii", "") for f in self.data_dir.glob("*.nii.gz")]
            )
            self.labels = None

    def __len__(self):
        return len(self.uids)

    def _load_nifti(self, uid):
        path = self.data_dir / f"{uid}.nii.gz"

        nii = nib.load(path)

        image = nii.get_fdata(dtype=np.float32)

        spacing = nii.header.get_zooms()[:3]

        return image, spacing

    def __getitem__(self, idx):

        uid = self.uids[idx]

        image, spacing = self._load_nifti(uid)

        # Normalisation simple
        image -= image.min()

        if image.max() > 0:
            image /= image.max()

        if self.transform is not None:

            try:
                image = self.transform(image, spacing)

            except TypeError:
                image = self.transform(image)

        if not torch.is_tensor(image):
            image = torch.from_numpy(image)

        # Ajout du canal (C,D,H,W)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        sample = {
            "uid": uid,
            "image": image,
        }

        if self.return_spacing:
            sample["spacing"] = torch.tensor(spacing, dtype=torch.float32)

        if self.labels is not None:
            sample["label"] = torch.tensor(
                self.labels[idx], dtype=torch.float32
            )

        return sample
    

