"""Plug in REAL data (NOT tested in the build sandbox - needs xarray + your files).

Download (free accounts):
  * CMEMS (marine.copernicus.eu): SST, sea-level anomaly (SLA), salinity
  * ERA5 (cds.climate.copernicus.eu): 10 m u/v wind
  * Target: CMEMS GLORYS12 reanalysis 'thetao' (temperature by depth) or ORAS5 / EN4
Clip all to the same box (e.g. lat 5-22, lon 78-95) and daily/weekly time steps.
Edit the FILE/VARIABLE names below to match what you downloaded.
"""
import numpy as np
import xarray as xr
from data import DEPTHS

FILES = dict(sst=("sst.nc", "analysed_sst"), sla=("sla.nc", "sla"),
             sss=("sss.nc", "so"), u=("wind.nc", "u10"), v=("wind.nc", "v10"),
             temp=("thetao.nc", "thetao"))


def load_real(files=FILES, depths=DEPTHS):
    d = {k: xr.open_dataset(f)[v] for k, (f, v) in files.items()}
    ref = d["sla"]                                   # common grid/time
    reg = {k: v.interp(latitude=ref.latitude, longitude=ref.longitude)
           .reindex(time=ref.time, method="nearest") for k, v in d.items() if k != "temp"}
    temp = d["temp"].sel(depth=list(depths), method="nearest") \
        .interp(latitude=ref.latitude, longitude=ref.longitude) \
        .reindex(time=ref.time, method="nearest")
    if float(reg["sst"].mean()) > 200:               # Kelvin -> Celsius
        reg["sst"] = reg["sst"] - 273.15
    t = ref.time.values
    doy = ref.time.dt.dayofyear.values - 1
    year = ref.time.dt.year.values - ref.time.dt.year.values.min()
    return dict(sst=reg["sst"].values, sla=reg["sla"].values, sss=reg["sss"].values,
                wind=np.hypot(reg["u"], reg["v"]).values,
                temp=temp.transpose("time", "depth", "latitude", "longitude").values,
                lat=ref.latitude.values, lon=ref.longitude.values,
                doy=doy, year=year, depths=np.array(depths))
