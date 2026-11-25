"""Inference helper: surface patch -> subsurface temperature profile."""
import numpy as np, joblib
from data import meta_features, CHANNELS

_bundle = None


def load(path="results/oceanembed_model.joblib"):
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(path)
    return _bundle


def predict_profile(patch_4ch, lat, lon, doy):
    """patch_4ch: array (4, p, p) in order sst, sla, sss, wind. Returns (depths, temps)."""
    b = load()
    x = b["xs"].transform(np.asarray(patch_4ch, dtype=np.float32).reshape(1, -1))
    ae = b["ae"]
    h = np.maximum(0, x @ ae.coefs_[0] + ae.intercepts_[0])
    e = np.maximum(0, h @ ae.coefs_[1] + ae.intercepts_[1])
    m = meta_features(dict(lat=np.array([lat]), lon=np.array([lon]), doy=np.array([doy])))
    z = np.c_[b["es"].transform(e), m]
    temps = b["ys"].inverse_transform(b["oe"].predict(z))[0]
    return b["depths"], temps
