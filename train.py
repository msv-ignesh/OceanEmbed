"""OceanEmbed: train + evaluate.  Run:  python train.py  [--real]"""
import sys, json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
import joblib
from data import make_synthetic, build_samples, meta_features, DEPTHS

SEED, EMB = 42, 32
PATCH = 5
OUT = "results"

# ------------------------------------------------------------------ data
if "--real" in sys.argv:
    from data_real import load_real
    ds = load_real(); DATA = "REAL"
else:
    ds = make_synthetic(); DATA = "SYNTHETIC"
X, meta, Y = build_samples(ds, patch=PATCH)
M = meta_features(meta)
yr = meta["year"]
n_years = yr.max() + 1
tr = yr < int(n_years * 0.7)                       # time-based split (no leakage)
va = (yr >= int(n_years * 0.7)) & (yr < int(n_years * 0.8))
te = yr >= int(n_years * 0.8)
print(f"[{DATA}] samples train/val/test = {tr.sum()}/{va.sum()}/{te.sum()}")

xs = StandardScaler().fit(X[tr]); ys = StandardScaler().fit(Y[tr])
Xs = xs.transform(X); Ys = ys.transform(Y)


def rmse(yt, yp):
    return np.sqrt(((yt - yp) ** 2).mean(0))


def evaluate(name, pred_scaled, store):
    p = ys.inverse_transform(pred_scaled)
    r = rmse(Y[te], p)
    store[name] = dict(rmse_per_depth=r.round(3).tolist(), rmse_mean=float(r.mean().round(3)),
                       mae_mean=float(np.abs(Y[te] - p).mean().round(3)),
                       r2_mean=float(r2_score(Y[te], p)))
    print(f"{name:34s} mean RMSE={r.mean():.3f}  R2={store[name]['r2_mean']:.3f}  per-depth={r.round(2)}")
    return p


def head(seed=SEED):
    return MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True, n_iter_no_change=8,
                        max_iter=150, random_state=seed, learning_rate_init=1e-3)


results, preds = {}, {}
t0 = time.time()

# ------------------------------------------------------------ baselines
b = Ridge(alpha=1.0).fit(np.c_[Xs[tr], M[tr]], Ys[tr])
preds["Linear (Ridge)"] = evaluate("Linear (Ridge)", b.predict(np.c_[Xs[te], M[te]]), results)

rf = RandomForestRegressor(40, max_depth=12, max_features=0.15, min_samples_leaf=10, n_jobs=-1, random_state=SEED)
rf.fit(np.c_[Xs[tr], M[tr]], Ys[tr])
preds["Random Forest"] = evaluate("Random Forest", rf.predict(np.c_[Xs[te], M[te]]), results)

mlp_raw = head().fit(np.c_[Xs[tr], M[tr]], Ys[tr])
preds["MLP on raw patches"] = evaluate("MLP on raw patches", mlp_raw.predict(np.c_[Xs[te], M[te]]), results)

# --------------------------------------------- OceanEmbed (2 stages)
# Stage 1: autoencoder learns a compact embedding of the surface patch
ae = MLPRegressor(hidden_layer_sizes=(128, EMB, 128), early_stopping=True, n_iter_no_change=8, max_iter=150,
                  random_state=SEED, learning_rate_init=1e-3).fit(Xs[tr], Xs[tr])


def embed(x):
    h = np.maximum(0, x @ ae.coefs_[0] + ae.intercepts_[0])
    return np.maximum(0, h @ ae.coefs_[1] + ae.intercepts_[1])        # (N, EMB)


E = embed(Xs)
ae_recon_r2 = r2_score(Xs[te], ae.predict(Xs[te]))
print(f"Stage 1 autoencoder reconstruction R2 (test): {ae_recon_r2:.3f}")

# Stage 2: embedding (+ lat/lon/season) -> temperature at all depths
es = StandardScaler().fit(E[tr])
Z = np.c_[es.transform(E), M]
oe = head().fit(Z[tr], Ys[tr])
preds["OceanEmbed (ours)"] = evaluate("OceanEmbed (ours)", oe.predict(Z[te]), results)

# ------------------------------------------------ ablation: which inputs matter
n = PATCH * PATCH
ch_idx = {c: np.arange(i, 4 * n, 4) for i, c in enumerate(["sst", "sla", "sss", "wind"])}
# X was raveled from (4, p, p) -> channel-major blocks
ch_idx = {c: np.arange(i * n, (i + 1) * n) for i, c in enumerate(["sst", "sla", "sss", "wind"])}
ablation = {}
for name, keep in {"SST only": ["sst"], "SST+SLA": ["sst", "sla"],
                   "SST+SLA+SSS": ["sst", "sla", "sss"], "All 4 channels": ["sst", "sla", "sss", "wind"],
                   "No SLA (SST+SSS+wind)": ["sst", "sss", "wind"]}.items():
    idx = np.concatenate([ch_idx[c] for c in keep])
    m_ = head().fit(np.c_[Xs[tr][:, idx], M[tr]], Ys[tr])
    p = ys.inverse_transform(m_.predict(np.c_[Xs[te][:, idx], M[te]]))
    ablation[name] = float(rmse(Y[te], p).mean().round(3))
    print(f"ablation {name:26s} mean RMSE={ablation[name]:.3f}")

# ------------------------------------------------------------------ save
results["_ablation_mean_rmse"] = ablation
results["_autoencoder_recon_r2"] = float(ae_recon_r2)
results["_data"] = DATA
json.dump(results, open(f"{OUT}/metrics.json", "w"), indent=2)
joblib.dump(dict(xs=xs, ys=ys, es=es, ae=ae, oe=oe, patch=PATCH, depths=DEPTHS),
            f"{OUT}/oceanembed_model.joblib")
print(f"done in {time.time()-t0:.0f}s")

# ----------------------------------------------------------------- plots
names = ["Linear (Ridge)", "Random Forest", "MLP on raw patches", "OceanEmbed (ours)"]
fig, ax = plt.subplots(figsize=(6, 4.5))
for nme in names:
    ax.plot(results[nme]["rmse_per_depth"], DEPTHS, marker="o", label=nme,
            lw=3 if "ours" in nme else 1.5)
ax.invert_yaxis(); ax.set_xlabel("Test RMSE (°C)"); ax.set_ylabel("Depth (m)")
ax.set_title(f"Error vs depth ({DATA} data)"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/rmse_vs_depth.png", dpi=150); plt.close()

fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
p = preds["OceanEmbed (ours)"]
for k, ax in enumerate(axs.ravel()):
    ax.scatter(Y[te][:, k], p[:, k], s=2, alpha=.3)
    lo, hi = Y[te][:, k].min(), Y[te][:, k].max()
    ax.plot([lo, hi], [lo, hi], "r--"); ax.set_title(f"{DEPTHS[k]} m")
    ax.set_xlabel("True °C"); ax.set_ylabel("Predicted °C")
plt.suptitle("OceanEmbed: predicted vs true (test set)"); plt.tight_layout()
plt.savefig(f"{OUT}/pred_vs_true.png", dpi=150); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
rng = np.random.default_rng(3)
for ax, i in zip(axs, rng.choice(np.where(te)[0], 3, replace=False)):
    j = np.where(np.where(te)[0] == i)[0][0]
    ax.plot(Y[i], DEPTHS, "k-o", label="True"); ax.plot(p[j], DEPTHS, "r--s", label="OceanEmbed")
    ax.invert_yaxis(); ax.set_xlabel("Temperature °C"); ax.grid(alpha=.3)
axs[0].set_ylabel("Depth (m)"); axs[0].legend(); plt.suptitle("Example vertical profiles (test)")
plt.tight_layout(); plt.savefig(f"{OUT}/profiles.png", dpi=150); plt.close()

sub = np.where(te)[0][:2500]
emb2 = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(E[sub])
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(emb2[:, 0], emb2[:, 1], c=meta["doy"][sub], cmap="twilight", s=4)
plt.colorbar(sc, label="Day of year"); ax.set_title("t-SNE of OceanEmbed embeddings")
plt.tight_layout(); plt.savefig(f"{OUT}/embedding_tsne.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.barh(list(ablation), list(ablation.values())); ax.set_xlabel("Mean test RMSE (°C)")
ax.set_title("Ablation: input variables"); plt.tight_layout()
plt.savefig(f"{OUT}/ablation.png", dpi=150); plt.close()
print("plots saved")
