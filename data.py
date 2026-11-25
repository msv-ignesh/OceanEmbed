"""Data layer for OceanEmbed.

Every dataset (synthetic or real) is a dict with:
  sst, sla, sss, wind : (T, H, W) surface satellite fields
  temp                : (T, D, H, W) subsurface temperature (target), NaN on land
  lat (H,), lon (W,), doy (T,), year (T,), depths (D,)
"""
import numpy as np
from scipy.ndimage import gaussian_filter

DEPTHS = np.array([0, 10, 50, 100, 200, 300])
CHANNELS = ["sst", "sla", "sss", "wind"]


def _field(rng, shape, sigma):
    f = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="nearest")
    return f / f.std()


def make_synthetic(years=10, H=20, W=20, seed=0):
    """Physically-inspired SYNTHETIC ocean (Bay of Bengal-like box).

    Idea: sea-level anomaly (SLA) tracks thermocline depth, so a warm/high-SLA
    eddy pushes warm water deeper. Part of the thermocline variability is
    hidden (not visible from the surface) -> irreducible error at depth.
    """
    rng = np.random.default_rng(seed)
    T = years * 365
    t = np.arange(T)
    doy = t % 365
    lat = np.linspace(5, 22, H)
    lon = np.linspace(78, 95, W)
    LAT = lat[None, :, None]
    season = 2 * np.pi * doy[:, None, None] / 365

    sig = (6, 2.5, 2.5)
    e_sst, e_sla, e_sss, e_wnd, e_hid = (_field(rng, (T, H, W), sig) for _ in range(5))

    sst = 29.5 - 0.12 * (LAT - 5) + 1.5 * np.sin(season - 1.0) + 0.6 * e_sst
    sla = 0.08 * e_sla + 0.05 * np.sin(season + 1.2)
    sss = 34.0 - 0.25 * (LAT - 5) + 0.5 * np.cos(season) + 0.3 * e_sss
    wind = np.abs(5 + 2 * np.cos(season - 0.5) + 1.2 * e_wnd)

    # thermocline depth (m): driven by SLA (observable) + hidden component
    D = 150 + 420 * sla + 0.35 * (sst - 29) * 10 + 14 * e_hid
    D = np.clip(D, 50, 300)
    w = 60.0
    T_deep = 10.0
    z = DEPTHS[None, :, None, None].astype(float)
    sst_b = sst[:, None]
    D_b = D[:, None]
    temp = T_deep + (sst_b - T_deep) * 0.5 * (1 - np.tanh((z - D_b) / w))
    temp = temp + rng.normal(0, 0.15, temp.shape)  # measurement noise

    return dict(sst=sst, sla=sla, sss=sss, wind=wind, temp=temp,
                lat=lat, lon=lon, doy=doy, year=t // 365, depths=DEPTHS)


def build_samples(ds, patch=5, per_day=12, seed=1):
    """Cut (patch x patch x 4-channel) windows around random ocean points.

    Returns X (N, 4*patch*patch), meta dict, Y (N, D).
    """
    rng = np.random.default_rng(seed)
    T, H, W = ds["sst"].shape
    r = patch // 2
    X, Y, lat, lon, doy, year = [], [], [], [], [], []
    stack = np.stack([ds[c] for c in CHANNELS], axis=1)  # (T, 4, H, W)
    for t in range(T):
        ii = rng.integers(r, H - r, per_day)
        jj = rng.integers(r, W - r, per_day)
        for i, j in zip(ii, jj):
            y = ds["temp"][t, :, i, j]
            if np.isnan(y).any():
                continue
            X.append(stack[t, :, i - r:i + r + 1, j - r:j + r + 1].ravel())
            Y.append(y)
            lat.append(ds["lat"][i]); lon.append(ds["lon"][j])
            doy.append(ds["doy"][t]); year.append(ds["year"][t])
    meta = dict(lat=np.array(lat), lon=np.array(lon),
                doy=np.array(doy), year=np.array(year))
    return np.array(X, dtype=np.float32), meta, np.array(Y, dtype=np.float32)


def meta_features(meta):
    """lat/lon (scaled) + seasonal sin/cos."""
    return np.c_[(meta["lat"] - 13.5) / 8.5, (meta["lon"] - 86.5) / 8.5,
                 np.sin(2 * np.pi * meta["doy"] / 365),
                 np.cos(2 * np.pi * meta["doy"] / 365)].astype(np.float32)
