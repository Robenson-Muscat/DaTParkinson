import SimpleITK as sitk
from pathlib import Path
import nibabel as nib
from nibabel.processing import resample_to_output
import numpy as np
import pandas as pd
import torch
#from torch.utils.data import Dataset
import torch.nn as nn
import os

import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy import ndimage
from concurrent.futures import ProcessPoolExecutor

import dask
from dask import delayed
from dask.diagnostics import ProgressBar




DATA_ROOT = Path("./data/")
NIFTI_DIR = DATA_ROOT / "niftis"
CSV_PATH = DATA_ROOT / "train_labels.csv"

CKPTS_DIR = Path("./ckpts/")
CKPTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(CSV_PATH)


TARGET_SPACING = 2.0
TARGET_SIZE = 112
DEFAULT_CROP_SIZE = 48




def find_nifti(uid, nifti_dir):
    """
    Find the NIfTI corresponding to a UID.

    Accepts:
        uid.nii
        uid.nii.gz
    """

    nifti_dir = Path(nifti_dir)

    candidates = [nifti_dir / f"{uid}.nii.gz", nifti_dir / f"{uid}.nii"]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"No NIfTI found for uid={uid} in {nifti_dir}")



def load_resample_nifti(path, target_spacing=TARGET_SPACING, interpolation_order=1):
    """
    Load a NIfTI, reorient it to RAS and resample to target spacing.

    Returns
    -------
    volume : np.ndarray
        Shape (X, Y, Z)
    affine : np.ndarray
        Affine corresponding to the resampled image.
    """

    img = nib.load(str(path))

    # Put all subjects in the same anatomical orientation
    img = nib.as_closest_canonical(img)

    # Resample to isotropic spacing
    img_resampled = resample_to_output(
        img,
        voxel_sizes=target_spacing,
        order=interpolation_order)

    volume = img_resampled.get_fdata(dtype=np.float32)

    return volume, img_resampled.affine



def _safe_volume(volume):
    """Replace invalid values and remove negative intensities."""
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(volume, 0.0)


def _crop_or_pad(volume, center, size):
    """Extract a fixed-size cube centered at center, padding with zeros."""
    size = np.broadcast_to(np.asarray(size, dtype=int), (3,))
    center = np.asarray(center, dtype=float)

    if np.any(size <= 0):
        raise ValueError("Crop size must be positive.")

    center = np.rint(center).astype(int)
    start = center - size // 2
    end = start + size

    output = np.zeros(tuple(size), dtype=np.float32)

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(volume.shape))

    # Handle crops that lie entirely outside the input volume.
    if np.any(src_end <= src_start):
        return output

    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    output[
        dst_start[0]:dst_end[0],
        dst_start[1]:dst_end[1],
        dst_start[2]:dst_end[2],
    ] = volume[
        src_start[0]:src_end[0],
        src_start[1]:src_end[1],
        src_start[2]:src_end[2],
    ]

    return output


def centro_testa(volume, percentile=99.5, threshold_ratio=0.30):
    """
    Estimate the head center using a binary foreground mask.

    The largest connected component reduces isolated noise.
    Returns the center in voxel coordinates.
    """
    volume = _safe_volume(volume)
    positive = volume[volume > 0]

    fallback = (np.asarray(volume.shape) - 1) / 2

    if positive.size < 100:
        return fallback

    reference = np.percentile(positive, percentile)
    threshold = threshold_ratio * reference

    if not np.isfinite(threshold) or threshold <= 0:
        return fallback

    mask = volume > threshold

    # Close small gaps without excessively expanding the foreground.
    mask = ndimage.binary_closing(mask, structure=ndimage.generate_binary_structure(3, 1),iterations=1)

    labels, n_labels = ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))

    if n_labels == 0:
        return fallback

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(np.argmax(counts))

    # Avoid trusting a tiny component in a malformed scan.
    if counts[largest_label] < 100:
        return fallback

    head_mask = labels == largest_label
    return np.asarray(ndimage.center_of_mass(head_mask))


def trova_hotspot(volume, q=99.5):
    """
    Estimate the bilateral hotspot center.

    Returns:
        center, spread, coordinates, threshold

    Coordinates and centers use voxel coordinates.
    """
    volume = _safe_volume(volume)

    positive = volume[volume > 0]
    fallback = (np.asarray(volume.shape) - 1) / 2

    if positive.size < 10:
        return fallback, np.zeros(3), np.empty((0, 3), int), 0.0

    threshold = np.percentile(positive, q)
    coords = np.argwhere(volume >= threshold)

    if len(coords) == 0:
        return fallback, np.zeros(3), coords, float(threshold)

    # Split along the first voxel axis after canonical orientation.
    mid = (volume.shape[0] - 1) / 2
    first = coords[coords[:, 0] < mid]
    second = coords[coords[:, 0] >= mid]

    centers = []

    if len(first):
        centers.append(first.mean(axis=0))
    if len(second):
        centers.append(second.mean(axis=0))

    # Give both sides equal influence when both are present.
    center = np.mean(centers, axis=0) if centers else coords.mean(axis=0)
    spread = coords.std(axis=0)

    return center, spread, coords, float(threshold)



def normalise_volume(volume):
    """
    Min-Max intensity normalization to [0, 1].

    Zero-valued background/padding remains zero.
    """
    volume = _safe_volume(volume)
    mask = volume > 0

    if not np.any(mask):
        return volume

    lo = np.min(volume[mask])
    hi = np.max(volume[mask])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)

    output = np.zeros_like(volume, dtype=np.float32)
    output[mask] = np.clip(
        (volume[mask] - lo) / (hi - lo),
        0.0,
        1.0)

    return output



def load_volume_mm(path, mm=TARGET_SPACING, side=TARGET_SIZE, return_metadata=False):
    """
    Load a NIfTI, canonicalize orientation, resample, and center
    a fixed-size cube on the estimated head center.
    """
    img = nib.load(path)

    original_orientation = nib.aff2axcodes(img.affine)
    img = nib.as_closest_canonical(img)
    ras_orientation = nib.aff2axcodes(img.affine)

    img_resampled = resample_to_output(img, voxel_sizes=(mm, mm, mm), order=1)

    volume = _safe_volume(img_resampled.get_fdata(dtype=np.float32))

    center = centro_testa(volume)
    volume = _crop_or_pad(volume, center, side)

    ## Preserve foreground contrast while retaining zero padding.
    volume  = normalise_volume(volume)

    if not return_metadata:
        return volume

    metadata = {
        "original_orientation": original_orientation,
        "orientation_after_canonical": ras_orientation,
        "resampled_shape": img_resampled.shape,
        "head_center": center,
        "voxel_size_mm": mm,
    }

    return volume, metadata


def crop_paziente(
    path,
    mm=TARGET_SPACING,
    side=TARGET_SIZE,
    crop=80,
    hotspot_percentile=99.5,
    return_metadata=False):
    
    """
    RAS-oriented DaTscan preprocessing:
        NIfTI -> RAS -> isotropic resampling
        -> head-centered volume -> striatal hotspot crop
        -> robust normalization.
    """

    if crop <= 0 or crop > side:
        raise ValueError("crop must be in the range 1..lato.")

    # carica_volume_mm now guarantees canonical RAS orientation
    volume, head_metadata = load_volume_mm(
        path,
        mm=mm,
        side=side,
        return_metadata=True,
    )

    # Estimate bilateral hotspot center
    center, spread, coords, threshold = trova_hotspot(volume, q=hotspot_percentile)
    #center, spread, coords, threshold = find_hotspot(volume,q=hotspot_percentile,)

    # Crop around the estimated striatum
    cropped = _crop_or_pad(volume, center=center, size=crop)

    # Normalize the crop
    #cropped = normalise_volume(cropped)

    metadata = {
        **head_metadata,
        "hotspot_center": center,
        "hotspot_spread": spread,
        "hotspot_threshold": threshold,
        "hotspot_voxel_count": len(coords),
        "crop_size": (crop, crop, crop),
        "final_shape": cropped.shape,
    }

    if return_metadata:
        return cropped, metadata

    return cropped


def preprocess_one(uid, label, nifti_dir, crop=48):
    uid = str(uid)
    path = find_nifti(uid, nifti_dir)

    volume, metadata = crop_paziente(path, mm=2.0,side=112, crop=crop, return_metadata=True)

    record = {
        "uid": uid,
        "label": label,
        "hotspot_center": metadata["hotspot_center"].tolist(),
        "hotspot_voxel_count": metadata["hotspot_voxel_count"]}

    return volume, record


def preprocess_dataframe(
    df,
    nifti_dir,
    crop=112,
    n_workers=None,
    batch_size=4,
    scheduler="threads",
    output_dir=None,
):
    """
    Prétraitement par lots avec Dask pour limiter
    la consommation mémoire.

    Parameters
    ----------
    df : pd.DataFrame
        Colonnes 'uid' et 'is_pathologic'.
    nifti_dir : str
        Répertoire des fichiers NIfTI.
    crop : int
        Taille du crop.
    n_workers : int or None
        Nombre de workers Dask.
    batch_size : int
        Nombre de volumes calculés simultanément.
    scheduler : str
        'threads' ou 'processes'.
    output_dir : str or None
        Si renseigné, enregistre chaque volume séparément
        en .npy pour éviter de garder tout le dataset
        en RAM.

    Returns
    -------
    volumes : np.ndarray or list
        Tableau empilé si output_dir=None, sinon liste
        des chemins des volumes enregistrés.
    metadata_df : pd.DataFrame
        Métadonnées de chaque volume.
    """

    if batch_size < 1:
        raise ValueError("batch_size doit être >= 1")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    records = []
    volumes = []
    volume_paths = []

    rows = list(zip(df["uid"], df["is_pathologic"]))

    if not rows:
        return np.empty((0,)), pd.DataFrame(
            columns=[
                "uid",
                "label",
                "hotspot_center",
                "hotspot_voxel_count",
            ]
        )

    #with ProgressBar():
    for start in tqdm(range(0, len(rows), batch_size), desc="Preprocessing"):
            batch = rows[start : start + batch_size]

            tasks = [delayed(preprocess_one)(uid, label, nifti_dir, crop) for uid, label in batch]

            # Calcul uniquement du lot courant
            results = dask.compute(*tasks, scheduler=scheduler, num_workers=n_workers)

            for volume, record in results:

                if output_dir is not None:
                    filename = f"{record['uid']}.npy"
                    path = os.path.join(output_dir, filename)

                    np.save(path, volume)
                    volume_paths.append(path)
                    del volume
                else:
                    volumes.append(volume)


            del results, tasks

    if output_dir is not None:
        return volume_paths

    return np.stack(volumes)

        


if __name__ == "__main__":
    volume_paths, metadata_df = preprocess_dataframe(
    df,
    nifti_dir="./data/niftis",
    crop=48,
    output_dir="./processed_volumes")