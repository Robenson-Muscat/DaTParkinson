# Creates features for voxels

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from preprocessing import load_volume_mm, VolumeStats
from concurrent.futures import ThreadPoolExecutor



DATA_ROOT = Path("./data/")
NIFTI_DIR = DATA_ROOT / "niftis"
CSV_PATH = DATA_ROOT / "train_labels.csv"



TARGET_SIZE = 112
QUANTILES = (10, 50, 70, 75, 90, 95, 97, 98, 99, 99.5, 99.9)



def find_hotspot(v, q=99.0, stats=None):
    """Return centroid, spread, coordinates and threshold of bright voxels."""
    if stats is not None and stats.v is v:
        return stats.hotspot(q)

    return VolumeStats(v).hotspot(q)

def crop_volume(v, mean, side):
    """Extract a centered cubic crop."""
    n = v.shape[0]
    h = side // 2
    c = np.clip(mean.astype(int), h, n - h)
    return v[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h]


def spatial_features(v, stats=None):
    """Extract hotspot, regional, asymmetry and spatial-grid features."""
    stats = stats if stats is not None else VolumeStats(v)
    n = v.shape[0]

    center, std, idx, threshold = stats.hotspot(99.0)
    f = list(center / n) + list(std / n)
    f += [len(idx) / v.size, float(v.ravel()[np.flatnonzero(
        v.ravel() >= threshold
    )].mean())]
    #f = list(mean / n) + list(std / n)
    #f += [len(idx) / v.size,float(v[idx[:, 0], idx[:, 1], idx[:, 2]].mean())]

    background = stats.percentile(50) + 1e-6
    
    for side in [n // 4, n // 3, n // 2]:              # three window sizes around the striatum
        r = crop_volume(v, center, side)
        p = np.percentile(r, [50, 75, 90, 95, 99])
        f += list(p / background)                           # ratios = binding ratio, measured in place
        f += [r.mean() / background, r.std() / background, float(r.max()) / background]
        

    r = crop_volume(v, center, n // 2)
    m = r.shape[0] // 2
    for axis in range(3):
        a = np.take(r, range(0, m), axis=axis)
        b = np.take(r, range(m, 2*m), axis=axis)
        for q in [50, 90, 99]:
            x, yq = np.percentile(a, q), np.percentile(b, q)
            f.append(abs(x - yq) / (x + yq + 1e-6))
        f.append(abs(a.mean() - b.mean()) / (a.mean() + b.mean() + 1e-6))
    g = r.shape[0] // 4
    q = r[:4*g, :4*g, :4*g]                            # trim to a multiple of 4 so the grid is always 4x4x4
    f += list((q.reshape(4, g, 4, g, 4, g).mean(axis=(1, 3, 5))).flatten() / background)
    return np.array(f, dtype=np.float32)



def stat(v, stats=None):
    if stats is not None and stats.v is v:
        p = [stats.percentile(q) for q in (10, 50, 90, 99)]
        mean, std = stats.mean, stats.std
    else:
        p = np.percentile(v, [10, 50, 90, 99])
        mean, std = v.mean(), v.std()

    return [*p, mean, std, p[3] / (p[1] + 1e-6)]

def volume_features(v, grid_size=4, stats=None):
    """Extract intensity, asymmetry, spatial-grid and hotspot features."""
    stats = stats if stats is not None else VolumeStats(v)
    n = v.shape[0]

    f = list(stat(v, stats))
    p = [stats.percentile(q)
         for q in (50, 75, 90, 95, 99, 99.5, 99.9)]
    f += list(p) + list(np.array(p[1:]) / (p[0] + 1e-6))

    z = (v - stats.mean) / (stats.std + 1e-6)
    f += [float((z**3).mean()), float((z**4).mean())]


    a, b = n // 4, 3 * n // 4
    central = v[a:b, a:b, a:b]
    f += list(stat(central))
    f += [central.mean() / (stats.mean + 1e-6), np.percentile(central, 99) / ((stats.percentile(50)) + 1e-6)]
    
    # Parkinson's onset is asymmetric, so left-right differences carry signal. The
    # anatomical orientation of the array is not assumed: all three axes are measured.
    
    midpoint = central.shape[0] // 2
    for axis in range(3):
        s1 = np.take(central, range(0, midpoint), axis=axis)
        s2 = np.take(central, range(midpoint, 2*midpoint), axis=axis)
        for q in [50, 90, 99]:
            x, yq = np.percentile(s1, q), np.percentile(s2, q)
            f.append(abs(x - yq) / (x + yq + 1e-6))

    while n % grid_size != 0:                                # the grid must divide the side
        grid_size -= 1
    cell_size = n // grid_size
    G = v.reshape(grid_size, cell_size, grid_size, cell_size, grid_size, cell_size).mean(axis=(1, 3, 5))
    f += list(G.flatten() / (v.mean() + 1e-6))
    for q in [95, 99]:                                     # where the bright signal sits
        idx = np.argwhere(v >= np.percentile(v, q))
        f += list(idx.mean(0) / n) + [float(idx.std(0).mean()) / n]
    #for q in (95, 99):
        #idx = stats.indices(q)
        #f += list(idx.mean(0) / n)
        #f += [float(idx.std(0).mean()) / n]
    return np.array(f, dtype=np.float32)
    


def axis_profile(region, axis, bin=8):
    """Compute mean intensity along one axis."""
    profile = region.mean(axis=tuple(a for a in range(3) if a != axis))
    boundaries = np.linspace(0, len(profile), bin + 1).astype(int)
    return [float(profile[boundaries[i]:boundaries[i+1]].mean()) for i in range(bin)]


def lobe_features(v, stats=None):
    """Extract axial profiles and intensity asymmetry between lobes."""
    stats = stats if stats is not None else VolumeStats(v)
    n = v.shape[0]

    center, _, idx, _ = stats.hotspot(99.5)
    r = crop_volume(v, center, n // 2)
    background = stats.percentile(50) + 1e-6

    # Conserver le reste de la fonction.
    f = []

    for axis in range(3):
        p = np.array(axis_profile(r, axis)) / background
        f += list(p)
        f += [float(p[:4].sum() / (p[4:].sum() + 1e-6))]   # front half over back half

    # The axis along which the bright voxels are most spread is the one separating the
    # two lobes; splitting at its median gives one lobe on each side.
    separation_axis = idx.std(0).argmax()
    split_position = np.median(idx[:, separation_axis])
    A, B = idx[idx[:, separation_axis] < split_position], idx[idx[:, separation_axis] >= split_position]
    val = []
    for L in (A, B):
        if len(L) < 5:
            val.append([0.0, 0.0, 0.0, 0.0])
            continue
        vi = v[L[:, 0], L[:, 1], L[:, 2]]
        val.append([float(vi.mean() / background), float(vi.max() / background),
                    len(L) / v.size, float(L.std(0).mean() / n)])
    a, b = np.array(val[0]), np.array(val[1])
    # Ordering by strength rather than by side makes the features invariant to which
    # hemisphere is affected, which we cannot read from the array anyway.
    stronger, weaker = np.maximum(a, b), np.minimum(a, b)
    f += list(stronger) + list(weaker)
    f += list((stronger - weaker) / (stronger + weaker + 1e-6))
    return np.array(f, dtype=np.float32)
    



def brain_background(v, stats=None):
    """Estimate tissue background while excluding air where possible."""
    stats = stats if stats is not None else VolumeStats(v)
    

    air_threshold = stats.percentile(50) * 0.5 + stats.mean * 0.25
    tissue = v[v > air_threshold]

    if tissue.size < 100:
        tissue = v[v > stats.percentile(70)]
    if tissue.size < 10:
        tissue = v.ravel()

    return float(np.median(tissue)) + 1e-6, tissue


def sbr(region, background):
    """Striatal binding ratio, the standard clinical quantity."""
    return float((region.mean() - background) / background)


def region_shape(idx, n):
    """Shape of a suprathreshold region from the eigenvalues of its covariance.

    A healthy striatum is elongated (comma-shaped); as uptake is lost it contracts
    towards a dot, so sphericity and elongation carry the same information a reader
    takes from the outline.
    """
    if len(idx) < 8:
        return [0.0] * 7
    c = idx - idx.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1]
    ev = np.maximum(ev, 1e-9)
    volume = len(idx)
    radius = (3 * volume / (4 * np.pi)) ** (1 / 3)
    return [volume / n**3,
            float(np.sqrt(ev[0]) / n), float(np.sqrt(ev[2]) / n),
            float(ev[2] / ev[0]),                      # sphericity: 1 sphere, 0 filament
            float(ev[1] / ev[0]),                      # flatness
            float(radius / n),
            float(np.sqrt(ev.sum()) / n)]


def clinical_morphological_features(v, stats=None):
    """Extract clinical, morphological, subregional and texture features."""
    stats = stats if stats is not None else VolumeStats(v)
    n = v.shape[0]


    background, tissue = brain_background(v, stats)
    f = [
        background,
        float(tissue.mean()),
        float(tissue.std()),
        float(tissue.size) / v.size]

    center, _, idx, _ = stats.hotspot(99.5)
    r = crop_volume(v, center, n // 2)

    for q in (97, 98, 99, 99.5, 99.9):
        indices = stats.indices(q)
        vi = v[indices[:, 0], indices[:, 1], indices[:, 2]]

        f.append(sbr(vi, background))
        f += region_shape(indices, n)

    f += [
        sbr(r, background),
        float(r.max() / background),
        float(np.percentile(r, 99) / background)]
    
    f += subregion_features(v, idx, background, n)
    f += texture(r)

    return np.nan_to_num(
        np.array(f, dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )

def texture(region, levels=16):
    """Compute histogram entropy, energy and local intensity differences."""
    quantized = np.clip((region / (region.max() + 1e-9) * (levels - 1)).astype(np.int16), 0, levels - 1)
    histo = np.bincount(quantized.flatten(), minlength=levels).astype(float)
    #histo /= histo.sum()
    probs = histo / histo.sum()
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
    g = [np.diff(region, axis=a) for a in range(3)]
    out = [entropy, float((probs ** 2).sum())]
    for gi in g:
        out += [float(np.abs(gi).mean()), float(gi.std())]
    diff = np.abs(np.diff(quantized, axis=0)).flatten()        # neighbour contrast, GLCM-like
    out += [float(diff.mean()), float(diff.std()), float((diff == 0).mean())]
    return out


def subregion_features(v, idx, background, n):
    """Split each lobe into thirds along its own long axis and compare them.

    The long axis runs caudate to putamen, so the three thirds approximate caudate,
    anterior putamen and posterior putamen. They are ordered by intensity rather than
    by position, because the anatomical direction of the array is not known.
    """
    separation_axis = idx.std(0).argmax()                          # axisarating the two lobes
    split_position = np.median(idx[:, separation_axis])
    out = []
    profiles = []
    for L in (idx[idx[:,separation_axis] < split_position], idx[idx[:, separation_axis] >= split_position]):
        if len(L) < 12:
            out += [0.0] * 8
            profiles.append([0.0, 0.0, 0.0])
            continue
        c = L - L.mean(0)
        vec = np.linalg.eigh(np.cov(c.T))[1][:, -1]
        t = c @ vec
        third_thresholds = np.percentile(t, [33.3, 66.7])
        parti = [L[t < third_thresholds[0]], L[(t >= third_thresholds[0]) & (t < third_thresholds[1])], L[t >= third_thresholds[1]]]
        val = [float(v[p[:, 0], p[:, 1], p[:, 2]].mean() / background) if len(p) > 3 else 0.0
               for p in parti]
        profiles.append(val)
        weakest, middle, strongest = sorted(val)
        out += val + [weakest / (strongest  + 1e-6),            # weakest over strongest third
                      middle / (strongest  + 1e-6),
                      float(np.std(val)),
                      len(L) / v.size,
                      float(np.sqrt(np.cov(c.T).trace()) / n)]
    a, b = np.array(profiles[0]), np.array(profiles[1])
    out += list(np.abs(a - b) / (a + b + 1e-6))        # asymmetry, third by third
    return out




def extract_features(volume_path, mm, size):
    v = load_volume_mm(volume_path, mm=mm, size=size)

    h = size // 2
    #vg = v.reshape(h, 2, h, 2, h, 2).mean(axis=(1, 3, 5))
    
    sv = VolumeStats(v)
    #sg = VolumeStats(vg)

    return np.concatenate([
        volume_features(v, grid_size=4, stats=sv),
        #volume_features(vg, grid_size=4, stats=sg),
        spatial_features(v, stats=sv),
        #spatial_features(vg, stats=sg),
        #lobe_features(v, stats=sv),
        #lobe_features(vg, stats=sg),
        clinical_morphological_features(v, stats=sv),
        #clinical_morphological_features(vg, stats=sg),
    ])

def build_df_features(volume_path):
    X = extract_features(
        os.path.join(NIFTI_DIR, volume_path),
        2.0,
        TARGET_SIZE)
    
    return volume_path, np.asarray(X).ravel()


if __name__=="__main__":
    
    df = pd.read_csv(CSV_PATH)
    list_files = [str(uid) + ".nii.gz" for uid in df["uid"]]
    list_labels = df["is_pathologic"].to_list()
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(build_df_features, list_files),
            total=len(list_files),
            desc="Feature Extraction"
        ))
    # Associate labels to files
    labels_dict = dict(zip(list_files, list_labels))
    
    features_tab = pd.DataFrame([x for _, x in results], index=[p.removesuffix(".nii.gz") for p, _ in results])
                                                                              
    
    features_tab.index.name = "file"
    features_tab["label"] = [labels_dict[p] for p in features_tab.index]
    features_tab.to_csv("features_tab.csv", index=True)
    features_tab.to_parquet("features_tab.parquet", index=True)


