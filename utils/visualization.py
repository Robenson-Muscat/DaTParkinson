import nibabel as nib
from nibabel.processing import resample_to_output
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from preprocessing import find_nifti, _safe_volume, _crop_or_pad, estimate_head_center
from features import find_hotspot


def visualize_preprocessing(
    uid,
    nifti_dir="./data/niftis",
    mm=2.0,
    lato=112,
    crop=48,
    hotspot_percentile=99.5,
    figsize=(15, 17),
):
    """
    Visualizes the preprocessing pipeline:
      1. Resampled volume
      2. Head-centered volume
      3. Striatum-centered volume
      4. Final normalized crop

    Columns: sagittal, coronal, axial.
    """

    # ---------------------------------------------------------
    # 1. Load the scan and prepare the resampled volume
    # ---------------------------------------------------------
    path = find_nifti(uid, nifti_dir)

    img = nib.load(str(path))
    img = nib.as_closest_canonical(img)

    img_resampled = resample_to_output(
        img,
        voxel_sizes=(mm, mm, mm),
        order=1,
    )

    resampled = _safe_volume(
        img_resampled.get_fdata(dtype=np.float32)
    )

    # ---------------------------------------------------------
    # 2. Estimate the head center and extract a fixed-size volume
    # ---------------------------------------------------------
    head_center = estimate_head_center(resampled)

    head_volume = _crop_or_pad(
        resampled,
        head_center,
        lato,
    )
    # head_volume = normalise_volume(head_volume)

    # ---------------------------------------------------------
    # 3. Locate hotspots and extract the final crop
    # ---------------------------------------------------------
    hotspot_center, spread, coords, threshold = find_hotspot(
        head_volume,
        q=hotspot_percentile)

    final_volume = _crop_or_pad(
        head_volume,
        hotspot_center,
        crop)

    final_volume = np.array(final_volume, dtype=np.float32, copy=True)

    # Min-max normalization
    v_min = final_volume.min()
    v_max = final_volume.max()

    if v_max > v_min:
        final_volume = (final_volume - v_min) / (v_max - v_min)
    else:
        final_volume = np.zeros_like(final_volume)

    # Hotspot coordinates in the final crop reference frame
    crop_center = np.asarray(final_volume.shape) // 2

    # ---------------------------------------------------------
    # 4. Extract the three orthogonal views
    # ---------------------------------------------------------
    def slices_at(volume, center):
        center = np.rint(center).astype(int)

        center = np.clip(
            center,
            0,
            np.asarray(volume.shape) - 1,
        )

        x, y, z = center

        return [
            np.rot90(volume[x, :, :]),   # Sagittal
            np.rot90(volume[:, y, :]),   # Coronal
            np.rot90(volume[:, :, z]),  # Axial
        ]

    # Center coordinates are specific to each volume.
    stages = [
        (
            "1. Resampled to 2 mm",
            resampled,
            slices_at(resampled, head_center),
        ),
        (
            "2. Head-centered volume",
            head_volume,
            slices_at(head_volume, np.array(head_volume.shape) / 2),
        ),
        (
            "3. Hotspot-centered volume",
            head_volume,
            slices_at(head_volume, hotspot_center),
        ),
        (
            f"4. Final normalized {crop}³ crop",
            final_volume,
            slices_at(final_volume, crop_center),
        ),
    ]

    # ---------------------------------------------------------
    # 5. Visualization
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        4, 3,
        figsize=figsize,
    )

    view_names = ["Sagittal", "Coronal", "Axial"]

    for row_idx, (stage_name, volume, images) in enumerate(stages):
        for col_idx, (view_name, image) in enumerate(
            zip(view_names, images)
        ):
            ax = axes[row_idx, col_idx]

            ax.imshow(image, cmap="hot", origin="upper")
            ax.set_title(f"{stage_name}\n{view_name}", fontsize=10)
            ax.axis("off")

    fig.suptitle(
        f"UID: {uid}\n",
        # f"Hotspot center: {np.round(hotspot_center, 1)} voxels | "
        # f"Threshold: {threshold:.4f} | "
        # f"Detected voxels: {len(coords)}",
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    print("File:", path)
    print("Resampled volume:", resampled.shape)
    print("Head center:", np.round(head_center, 2))
    print("Hotspot center:", np.round(hotspot_center, 2))
    print("Hotspot spread:", np.round(spread, 2))
    print("Hotspot threshold:", threshold)
    print("Number of hotspot voxels:", len(coords))
    print("Final crop:", final_volume.shape)

    return final_volume

if __name__=="__main__":
    visualize_preprocessing("xaji0y6d")
