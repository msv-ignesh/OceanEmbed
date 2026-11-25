"""Demo:  streamlit run app.py   (pick a date + location, see predicted profile)"""
import numpy as np, streamlit as st, matplotlib.pyplot as plt
from data import make_synthetic
from predict import predict_profile

st.title("🌊 OceanEmbed – ocean subsurface temperature from satellite surface data")


@st.cache_data
def get_ds():
    return make_synthetic()


ds = get_ds()
T, H, W = ds["sst"].shape
day = st.slider("Day index (years 9-10 = unseen test period)", 365 * 8, T - 1, 365 * 8 + 100)
i = st.slider("Latitude index", 2, H - 3, 10)
j = st.slider("Longitude index", 2, W - 3, 10)
r = 2
patch = np.stack([ds[c][day, i - r:i + r + 1, j - r:j + r + 1] for c in ["sst", "sla", "sss", "wind"]])
d, p = predict_profile(patch, ds["lat"][i], ds["lon"][j], ds["doy"][day])
true = ds["temp"][day, :, i, j]
fig, ax = plt.subplots(figsize=(4.5, 5))
ax.plot(true, d, "k-o", label="True"); ax.plot(p, d, "r--s", label="OceanEmbed")
ax.invert_yaxis(); ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Depth (m)")
ax.grid(alpha=.3); ax.legend()
st.pyplot(fig)
st.write(f"Surface SST {patch[0, r, r]:.2f} °C | SLA {patch[1, r, r]*100:.1f} cm")
