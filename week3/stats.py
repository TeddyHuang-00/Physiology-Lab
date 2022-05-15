from matplotlib.colors import Colormap
from numpy.lib.function_base import average
import streamlit as slt
import os
from typing import List
import pandas as pd
import chardet
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dataFile = "./data/NoResponseLapse-Peak.txt"
with open(dataFile) as f:
    rawData = pd.read_table(f, index_col=0, low_memory=False)
    rawData = rawData.sort_index()
    # rawData = rawData.iloc[:, 0:3]

# rawData.iloc[rawData.事件.first_valid_index() :, 0]

fig = plt.figure(figsize=(7, 7))
rp = rawData.index
cf = rawData.iloc[:, 0]
plt.scatter(
    rp, cf, label="SC to RP", alpha=0.75,
)
plt.plot(rp, cf, alpha=0.6)
plt.vlines(0.7, min(cf), max(cf), "tab:grey", alpha=0.5)
plt.vlines(0.75, min(cf), max(cf), "tab:grey", alpha=0.5)
plt.fill_between(
    np.array([0.7, 0.75]),
    np.array([min(cf), min(cf)]),
    np.array([max(cf), max(cf)]),
    color="tab:grey",
    alpha=0.3,
    label="Interval of RP's end",
)
plt.hlines(
    0.68,
    min(rp),
    max(rp),
    "tab:purple",
    alpha=0.75,
    label="Typical value of normal CF",
)
plt.xlabel("Refractory period / RP (s)")
plt.ylabel("Contractility force / CF (g)")
plt.title("Stimulated Contractility Force(CF) to Refractory Period(RP) length")
plt.legend()
# plt.show()
slt.pyplot(fig)
