# Creates features for voxels

import numpy as np




def trova_hotspot(v, q=99.0):
    """Centroid, spread and coordinates of the brightest voxels, i.e. the striatum."""
    soglia = np.percentile(v, q)
    idx = np.argwhere(v >= soglia)
    centro = idx.mean(0)
    disp = idx.std(0)
    return centro, disp, idx, soglia


def ritaglia(v, centro, lato):
    n = v.shape[0]
    h = lato // 2
    c = np.clip(centro.astype(int), h, n - h)
    return v[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h]


def feature_allineate(v):
    n = v.shape[0]
    centro, disp, idx, soglia = trova_hotspot(v)
    f = list(centro / n) + list(disp / n)
    f += [len(idx) / v.size, float(v[v >= soglia].mean())]

    fondo = np.percentile(v, 50) + 1e-6                # background reference
    for lato in [n // 4, n // 3, n // 2]:              # three window sizes around the striatum
        r = ritaglia(v, centro, lato)
        p = np.percentile(r, [50, 75, 90, 95, 99])
        f += list(p / fondo)                           # ratios = binding ratio, measured in place
        f += [r.mean() / fondo, r.std() / fondo, float(r.max()) / fondo]

    r = ritaglia(v, centro, n // 2)
    m = r.shape[0] // 2
    for asse in range(3):
        a = np.take(r, range(0, m), axis=asse)
        b = np.take(r, range(m, 2*m), axis=asse)
        for q in [50, 90, 99]:
            x, yq = np.percentile(a, q), np.percentile(b, q)
            f.append(abs(x - yq) / (x + yq + 1e-6))
        f.append(abs(a.mean() - b.mean()) / (a.mean() + b.mean() + 1e-6))
    g = r.shape[0] // 4
    q = r[:4*g, :4*g, :4*g]                            # trim to a multiple of 4 so the grid is always 4x4x4
    f += list((q.reshape(4, g, 4, g, 4, g).mean(axis=(1, 3, 5))).flatten() / fondo)
    return np.array(f, dtype=np.float32)



def stat(v):
    p = np.percentile(v, [10, 50, 90, 99])
    return [p[0], p[1], p[2], p[3], v.mean(), v.std(), p[3] / (p[1] + 1e-6)]


def feature_volume(v, griglia=4):
    """Features of a cubic normalised volume, at whatever side length it has."""
    n = v.shape[0]
    f = list(stat(v))
    p = np.percentile(v, [50, 75, 90, 95, 99, 99.5, 99.9])
    f += list(p) + list(p[1:] / (p[0] + 1e-6))

    z = (v - v.mean()) / (v.std() + 1e-6)
    f += [float((z**3).mean()), float((z**4).mean())]      # skewness and kurtosis

    a, b = n // 4, 3 * n // 4
    c = v[a:b, a:b, a:b]
    f += list(stat(c))
    f += [c.mean() / (v.mean() + 1e-6), np.percentile(c, 99) / (np.percentile(v, 50) + 1e-6)]

    # Parkinson's onset is asymmetric, so left-right differences carry signal. The
    # anatomical orientation of the array is not assumed: all three axes are measured.
    m = c.shape[0] // 2
    for asse in range(3):
        s1 = np.take(c, range(0, m), axis=asse)
        s2 = np.take(c, range(m, 2*m), axis=asse)
        for q in [50, 90, 99]:
            x, yq = np.percentile(s1, q), np.percentile(s2, q)
            f.append(abs(x - yq) / (x + yq + 1e-6))



# Feature_lobes



def profilo(r, asse, bin=8):
    m = r.mean(axis=tuple(a for a in range(3) if a != asse))
    idx = np.linspace(0, len(m), bin + 1).astype(int)
    return [float(m[idx[i]:idx[i+1]].mean()) for i in range(bin)]


def feature_lobi(v):
    n = v.shape[0]
    centro, _, idx, _ = trova_hotspot(v, q=99.5)
    r = ritaglia(v, centro, n // 2)
    fondo = np.percentile(v, 50) + 1e-6
    f = []

    for asse in range(3):
        p = np.array(profilo(r, asse)) / fondo
        f += list(p)
        f += [float(p[:4].sum() / (p[4:].sum() + 1e-6))]   # front half over back half

    # The axis along which the bright voxels are most spread is the one separating the
    # two lobes; splitting at its median gives one lobe on each side.
    sep = idx.std(0).argmax()
    taglio = np.median(idx[:, sep])
    A, B = idx[idx[:, sep] < taglio], idx[idx[:, sep] >= taglio]
    val = []
    for L in (A, B):
        if len(L) < 5:
            val.append([0.0, 0.0, 0.0, 0.0])
            continue
        vi = v[L[:, 0], L[:, 1], L[:, 2]]
        val.append([float(vi.mean() / fondo), float(vi.max() / fondo),
                    len(L) / v.size, float(L.std(0).mean() / n)])
    a, b = np.array(val[0]), np.array(val[1])
    # Ordering by strength rather than by side makes the features invariant to which
    # hemisphere is affected, which we cannot read from the array anyway.
    forte, debole = np.maximum(a, b), np.minimum(a, b)
    f += list(forte) + list(debole)
    f += list((forte - debole) / (forte + debole + 1e-6))
    return np.array(f, dtype=np.float32)
    while n % griglia != 0:                                # the grid must divide the side
        griglia -= 1
    g = n // griglia
    G = v.reshape(griglia, g, griglia, g, griglia, g).mean(axis=(1, 3, 5))
    f += list(G.flatten() / (v.mean() + 1e-6))
    for q in [95, 99]:                                     # where the bright signal sits
        idx = np.argwhere(v >= np.percentile(v, q))
        f += list(idx.mean(0) / n) + [float(idx.std(0).mean()) / n]
    return np.array(f, dtype=np.float32)

#Features_clinical

def fondo_cerebrale(v):
    """Background level: the median of tissue, not of air.

    Air occupies most of the field of view, so a plain median of the volume would
    measure nothing. The threshold separates head from air roughly, with two fallbacks
    for volumes where it cuts too much.
    """
    soglia_aria = np.percentile(v, 50) * 0.5 + v.mean() * 0.25
    dentro = v[v > soglia_aria]
    if dentro.size < 100:
        dentro = v[v > np.percentile(v, 70)]
    if dentro.size < 10:
        dentro = v.ravel()
    return float(np.median(dentro)) + 1e-6, dentro


def sbr(regione, fondo):
    """Striatal binding ratio, the standard clinical quantity."""
    return float((regione.mean() - fondo) / fondo)


def forma(idx, n):
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
    raggio = (3 * volume / (4 * np.pi)) ** (1 / 3)
    return [volume / n**3,
            float(np.sqrt(ev[0]) / n), float(np.sqrt(ev[2]) / n),
            float(ev[2] / ev[0]),                      # sphericity: 1 sphere, 0 filament
            float(ev[1] / ev[0]),                      # flatness
            float(raggio / n),
            float(np.sqrt(ev.sum()) / n)]


def feature_v6(v):
    n = v.shape[0]
    fondo, dentro = fondo_cerebrale(v)
    f = [fondo, float(dentro.mean()), float(dentro.std()), float(dentro.size) / v.size]

    centro, _, idx, _ = trova_hotspot(v, q=99.5)
    r = ritaglia(v, centro, n // 2)
    for q in [97, 98, 99, 99.5, 99.9]:                 # binding ratio at several thresholds
        m = v >= np.percentile(v, q)
        f.append(sbr(v[m], fondo))
        f += forma(np.argwhere(m), n)
    f += [sbr(r, fondo), float(r.max() / fondo), float(np.percentile(r, 99) / fondo)]
    f += sotto_regioni(v, idx, fondo, n)
    f += texture(r)
    # A single NaN would poison the whole feature matrix and every model downstream.
    return np.nan_to_num(np.array(f, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def texture(r, livelli=16):
    q = np.clip((r / (r.max() + 1e-9) * (livelli - 1)).astype(np.int16), 0, livelli - 1)
    p = np.bincount(q.flatten(), minlength=livelli).astype(float)
    p /= p.sum()
    entropia = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    g = [np.diff(r, axis=a) for a in range(3)]
    out = [entropia, float((p ** 2).sum())]
    for gi in g:
        out += [float(np.abs(gi).mean()), float(gi.std())]
    diff = np.abs(np.diff(q, axis=0)).flatten()        # neighbour contrast, GLCM-like
    out += [float(diff.mean()), float(diff.std()), float((diff == 0).mean())]
    return out


def sotto_regioni(v, idx, fondo, n):
    """Split each lobe into thirds along its own long axis and compare them.

    The long axis runs caudate to putamen, so the three thirds approximate caudate,
    anterior putamen and posterior putamen. They are ordered by intensity rather than
    by position, because the anatomical direction of the array is not known.
    """
    sep = idx.std(0).argmax()                          # axis separating the two lobes
    taglio = np.median(idx[:, sep])
    out = []
    profili = []
    for L in (idx[idx[:, sep] < taglio], idx[idx[:, sep] >= taglio]):
        if len(L) < 12:
            out += [0.0] * 8
            profili.append([0.0, 0.0, 0.0])
            continue
        c = L - L.mean(0)
        vec = np.linalg.eigh(np.cov(c.T))[1][:, -1]
        t = c @ vec
        terzi = np.percentile(t, [33.3, 66.7])
        parti = [L[t < terzi[0]], L[(t >= terzi[0]) & (t < terzi[1])], L[t >= terzi[1]]]
        val = [float(v[p[:, 0], p[:, 1], p[:, 2]].mean() / fondo) if len(p) > 3 else 0.0
               for p in parti]
        profili.append(val)
        est, med, altro = sorted(val)
        out += val + [est / (altro + 1e-6),            # weakest over strongest third
                      med / (altro + 1e-6),
                      float(np.std(val)),
                      len(L) / v.size,
                      float(np.sqrt(np.cov(c.T).trace()) / n)]
    a, b = np.array(profili[0]), np.array(profili[1])
    out += list(np.abs(a - b) / (a + b + 1e-6))        # asymmetry, third by third
    return out



def feature_scala(percorso, mm, lato):
    """672 engineered features at one physical scale, plus the same at half resolution.

    The feature blocks are nested prefixes: the first 518 columns are what the linear
    branches consume, all 672 go to the gradient-boosted ones.
    """
    v = carica_volume_mm(percorso, mm=mm, lato=lato)
    h = lato // 2
    vg = v.reshape(h, 2, h, 2, h, 2).mean(axis=(1, 3, 5))
    return np.concatenate([feature_volume(v, griglia=4), feature_volume(vg, griglia=4),
                           feature_allineate(v), feature_allineate(vg),
                           feature_lobi(v), feature_lobi(vg),
                           feature_v6(v), feature_v6(vg)])
