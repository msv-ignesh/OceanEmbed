# OceanEmbed — Satellite Embedding-Based Deep Learning Framework for Subsurface Ocean Temperature Reconstruction

Predict the temperature at depth (0, 10, 50, 100, 200, 300 m) from what satellites see at the surface
(SST, sea-level anomaly, salinity, wind).

## Method
1. **Stage 1 – Embedding:** an autoencoder (neural net, 100 -> 128 -> **32** -> 128 -> 100) compresses each
   5x5 patch of 4 surface channels into a 32-d "OceanEmbed" vector (test reconstruction R² = 0.999).
2. **Stage 2 – Reconstruction:** a neural-net regressor maps [embedding + lat, lon, sin/cos(day-of-year)]
   -> temperature at 6 depths.
3. **Baselines:** Ridge regression, Random Forest, MLP on raw patches (no embedding).
4. **Evaluation:** time-based split (train yrs 1-7 / val 8 / test 9-10), RMSE/MAE/R² per depth, input-variable ablation, t-SNE of embeddings.

## Files
| File | Purpose |
|---|---|
| `data.py` | synthetic ocean generator + patch sampling |
| `data_real.py` | loader for real CMEMS/ERA5/GLORYS NetCDF files (untested, edit file names) |
| `train.py` | full pipeline: baselines, OceanEmbed, ablation, plots, saves model |
| `predict.py`, `app.py` | inference + Streamlit demo |
| `results/` | metrics.json, plots, trained model, log |

## Run
```
pip install -r requirements.txt
python train.py          # ~2-3 min on CPU, synthetic data
streamlit run app.py     # demo
python train.py --real   # after configuring data_real.py
```

## Results (SYNTHETIC data, test years)
| Model | Mean RMSE (°C) | R² |
|---|---|---|
| Ridge | 0.677 | 0.774 |
| Random Forest | 0.547 | 0.848 |
| MLP on raw patches | 0.595 | 0.823 |
| **OceanEmbed (32-d)** | 0.615 | 0.813 |

Ablation (mean RMSE): SST only 1.31 | SST+SLA 0.58 | SST+SLA+SSS 0.59 | all 4: 0.60 | without SLA 1.44.
**Takeaways:** (i) sea-level anomaly is by far the most informative input; (ii) error peaks near the
thermocline (100-200 m); (iii) a 32-d embedding keeps almost all of the accuracy (0.615 vs 0.595)
while compressing the input ~3x. On this synthetic data OceanEmbed does NOT beat the raw-input models.
