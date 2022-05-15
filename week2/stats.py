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


def GB2UTF(rootPath: str):
    fileList = []
    for fileName in os.listdir(rootPath):
        fileList.append(rootPath + fileName)
    for filePath in fileList:
        with open(filePath, mode="rb") as f:
            data = f.read()
            encodingType = chardet.detect(data)
            if encodingType["encoding"] == "utf-8":
                continue
        with open(filePath, mode="r", encoding=encodingType["encoding"]) as f:
            data = f.read()
        with open(
            filePath[:-4] + "_new" + filePath[-4:], mode="w", encoding="utf-8"
        ) as f:
            f.write(data)


@slt.cache
def readCleanData(cleanDataPath: str):
    cleanFileList = [cleanDataPath + fileName for fileName in os.listdir(cleanDataPath)]
    cleanDataList = []
    mV = 10
    for cleanFileName in cleanFileList:
        mV += 10
        with open(cleanFileName) as cf:
            df = pd.read_table(cf, header=2)
            df = df.iloc[df.事件.first_valid_index() :, 0]
            df.name = f"{mV}"
            cleanDataList.append(df)
    return cleanDataList


def sliceData(cleanData: pd.DataFrame, setoff: int, roundNum: int):
    seqLen = (len(cleanData) + setoff) // roundNum
    perRunData = []
    for i in range(roundNum):
        perRunData.append(cleanData[i * seqLen + setoff : (i + 1) * seqLen])
    return perRunData


def logistic_increase_function(t, K, P0, r, t0):
    # """
    # t: time
    # P0: initial value
    # K: capacity
    # r: increase rate
    # t0: initial time
    # """
    expValue = np.exp(r * (t - t0))
    return (K * expValue * P0) / (K + (expValue - 1) * P0)


def simplified_logistic_function(t, K, b, t0):
    expValue = np.exp((t - t0) / b)
    return K * expValue / (1 + expValue)


@slt.cache(suppress_st_warning=True)
def logisticRegression(x: np.array, y: np.array):
    popt, pcov = curve_fit(logistic_increase_function, x, y)
    P_predict = logistic_increase_function(x, *popt)
    return popt, P_predict


@slt.cache(suppress_st_warning=True)
def simplifiedLogisticRegression(x: np.array, y: np.array):
    popt, pcov = curve_fit(simplified_logistic_function, x, y)
    P_predict = simplified_logistic_function(x, *popt)
    return popt, P_predict


def power_decrease_function(x, coEf, pow, C):
    return coEf * np.power(x, -pow) + C


# def power_decrease_function(x, coEf, pow, C, scale):
#     return coEf * np.power(x / scale, -pow) + C


@slt.cache(suppress_st_warning=True)
def powerRegression(x: np.array, y: np.array):
    popt, pcov = curve_fit(power_decrease_function, x, y)
    P_predict = power_decrease_function(x, *popt)
    return popt, P_predict


def calcRSquared(x: np.array, y: np.array, popt, func: callable):
    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    RSquared = 1 - (ss_res / ss_tot)
    return RSquared


slt.title("Action potential")

dataSet = readCleanData("./data/ExpB_clean/")
dataPerRun = sliceData(dataSet[0], 100, 50)
maxValues = np.array([max(x) for x in dataPerRun])
# dataPerRun
# maxValues
mVoltage = np.array([(i + 2) * 10 for i in range(50)])
popt, P_predict = logisticRegression(mVoltage, maxValues)
RSquared = calcRSquared(mVoltage, maxValues, popt, logistic_increase_function)
slt.write("$R^2$:", RSquared)
slt.write(
    f"K:capacity {popt[0]}\n\nP0:initial_value {popt[1]}\n\nr:increase_rate {popt[2]}\n\nt0: start_time {popt[3]}"
)
fig = plt.figure(figsize=(7, 7))
plt.scatter(mVoltage, maxValues, alpha=0.75, label="Peak amplitude in each run")
plt.plot(mVoltage, P_predict, label="Logistic regression", color="tab:orange")
plt.xlabel("Stimulus strength (mV)")
plt.ylabel("Peak Amplitude (mV)")
plt.title("Nerve's peak amplitude to stimulus strength (10ms each)")
x0 = 297
y0 = (popt[1] + popt[0]) / 2
K = popt[0]
r = popt[2]
plt.hlines((popt[1] + popt[0]) / 2, 20, 510, "grey", alpha=0.5)
plt.vlines(x0, 0, 9, "grey", alpha=0.5)
x1 = 373
x2 = 222
plt.plot(
    [x2, x1],
    [y0 + r * y0 * (K - y0) / K * (x2 - x0), y0 + r * y0 * (K - y0) / K * (x1 - x0)],
    "tab:green",
    alpha=0.75,
)
plt.text(x0 + 10, y0 + 0.2, f"b = {r*y0*(K-y0)/K :>.3f}")
plt.text(x0 + 10, 0.15, f"x0 = {x0}")
plt.legend()
slt.pyplot(fig)
slt.write(K)
slt.write(r * y0 * (K - y0) / K)

popt, P_predict = simplifiedLogisticRegression(mVoltage, maxValues)
RSquared = calcRSquared(mVoltage, maxValues, popt, simplified_logistic_function)
slt.write("$R^2$:", RSquared)
slt.write("$R^2$:", RSquared)
slt.write(
    f"K:capacity {popt[0]}\n\nb:increase_rate {popt[1]}\n\nt0: start_time {popt[2]}"
)
fig = plt.figure(figsize=(7, 7))
plt.scatter(mVoltage, maxValues, alpha=0.75, label="Peak amplitude in each run")
plt.plot(mVoltage, P_predict, label="Logistic regression", color="tab:orange")
plt.xlabel("Stimulus strength (mV)")
plt.ylabel("Peak Amplitude (mV)")
plt.title("Nerve's peak amplitude to stimulus strength (10ms each)")
x0 = 297
slt.write(popt[2])
y0 = popt[0] / 2
K = popt[0]
b = popt[1]
plt.hlines(y0, 20, 510, "grey", alpha=0.5)
plt.vlines(x0, 0, 9, "grey", alpha=0.5)
x1 = 373
x2 = 222
plt.plot(
    [x2, x1],
    [y0 + y0 * (K - y0) / b / K * (x2 - x0), y0 + y0 * (K - y0) / b / K * (x1 - x0)],
    "tab:green",
    alpha=0.75,
)
plt.text(x0 + 10, y0 + 0.2, f"b = {y0*(K-y0)/b/K :>.3f}")
plt.text(x0 + 10, 0.15, f"x0 = {x0}")
plt.legend()
slt.pyplot(fig)
slt.write(K)
slt.write(y0 * (K - y0) / b / K)

ExpECleanDataRootPath = "./data/ExpE_clean/"
expEData = readCleanData(ExpECleanDataRootPath)
fig = plt.figure(figsize=(7, 7))
plt.plot(expEData[0])
plt.plot(expEData[1])
slt.pyplot(fig)
len(expEData[0]), len(expEData[1])
# expEData[0]
double = sliceData(expEData[0], 250, 7)
single = sliceData(expEData[1], 250, 10)
# double, single
sd = np.average(double[:5], axis=0)
ss = np.average(single[0:3], axis=0)
sd = sd[:1540]
ss = ss[:1540]
sd, ss
sdH = 586
sdPS = 745
sdPE = 954
sdMax = 865
sdMin = 1030
sdNE = 1385
ssH = 587
ssPS = 777
ssPE = 1128
ssMax = 942
ssMin = 1235
ssNE = 1539
tt = [(i + 1) * 0.005 for i in range(1540)]
fig = plt.figure(figsize=(7, 7))
plt.plot(tt, sd, label="Biphasic action potential (BAP)")
plt.fill_between(tt[sdPS : sdPE + 1], sd[sdPS : sdPE + 1], 0, alpha=0.25)
plt.scatter(
    [tt[sdPS], tt[sdPE], tt[sdMax], tt[sdMin], tt[sdH], tt[sdNE]],
    [0, 0, sd[sdMax], sd[sdMin], sd[sdH], 0],
    c="tab:blue",
    alpha=0.5,
)
# plt.vlines(tt[sdMax], 0, sd[sdMax], "tab:blue", alpha=0.5)
# plt.vlines(tt[sdMin], sd[sdMin], 0, "tab:blue", alpha=0.5)
# plt.vlines(tt[sdH], sd[sdH], 0, "tab:blue", alpha=0.5)
plt.fill_betweenx([min(sd), max(sd)], tt[sdH], tt[sdPS], color="tab:blue", alpha=0.25)
plt.text(tt[sdMax] + 0.1, sd[sdMax], f"BAP max: {sd[sdMax] :>.3f}")
plt.text(tt[sdMin] + 0.1, sd[sdMin] - 0.25, f"BAP min: {sd[sdMin] :>.3f}")
plt.plot(tt, ss, label="Monophasic action potential (MAP)")
plt.fill_between(tt[ssPS : ssPE + 1], ss[ssPS : ssPE + 1], 0, alpha=0.25)
plt.scatter(
    [tt[ssPS], tt[ssPE], tt[ssMax], tt[ssMin], tt[ssH]],
    [0, 0, ss[ssMax], ss[ssMin], ss[ssH]],
    c="tab:orange",
    alpha=0.5,
)
# plt.vlines(tt[ssMax], 0, ss[ssMax], "tab:orange", alpha=0.5)
# plt.vlines(tt[ssMin], ss[ssMin], 0, "tab:orange", alpha=0.5)
# plt.vlines(tt[ssH], ss[ssH], 0, "tab:orange", alpha=0.5)
plt.fill_betweenx([min(sd), max(sd)], tt[ssH], tt[ssPS], color="tab:orange", alpha=0.25)
plt.text(tt[ssMax] + 0.1, ss[ssMax], f"MAP max: {ss[ssMax] :>.3f}")
plt.text(tt[ssMin] - 0.2, ss[ssMin] - 0.6, f"MAP min: {ss[ssMin] :>.3f}")
plt.text(tt[ssH] - 2, 2, "Hibernation period")
plt.hlines(0, min(tt), max(tt), "tab:grey")
plt.legend()
plt.title("Comparison between bi- & mono-phasic action potential")
plt.xlabel("Time (ms)")
plt.ylabel("Detected signal voltage (mV)")
slt.pyplot(fig)

fig = plt.figure(figsize=(7, 7))
maxValues = []
for each in double:
    maxValues.append(max(each))
maxValues
plt.scatter([i + 1 for i in range(7)], maxValues, label="BAP (first)")
maxValues = []
for each in single:
    maxValues.append(max(each))
maxValues
plt.scatter([i + 1 for i in range(7, 17)], maxValues, label="MAP (last)")
plt.legend()
plt.xlabel("No. of test run (by time)")
plt.ylabel("Peak amplitude (mV)")
plt.title("Variation in peak amplitude along with time (test run)")
slt.pyplot(fig)

ThresholdAgainstTimeWidth = {
    0.05: 0.31,
    0.06: 0.29,
    0.08: 0.28,
    0.12: 0.17,
    0.16: 0.14,
    0.24: 0.12,
    0.32: 0.09,
    0.48: 0.08,
    0.64: 0.07,
    0.80: 0.07,
    1.20: 0.06,
    1.60: 0.06,
}
ThresholdAgainstTimeWidth
fig = plt.figure(figsize=(7, 7))
timeWidth = np.array(list(ThresholdAgainstTimeWidth.keys()))
threshold = np.array(list(ThresholdAgainstTimeWidth.values()))
popt, P_predict = powerRegression(timeWidth, threshold)
plt.scatter(
    timeWidth, threshold, alpha=0.75, label="Threshold stimulus of action potential"
)
plt.plot(timeWidth, P_predict, color="tab:orange", label="Power function regression")
plt.hlines(popt[2], min(timeWidth), max(timeWidth), color="tab:grey", alpha=0.5)
plt.text(0.1, popt[2] + 0.01, f"Calculated minimum threshold stimulus {popt[2]:>.3f}")
plt.xlabel("Stimulus duration (ms)")
plt.ylabel("Threshold stimulus (V)")
plt.title("Threshold stimulus to stimulus duration")
plt.legend()
# P_predict
RSquared = calcRSquared(timeWidth, threshold, popt, power_decrease_function)
slt.write("$R^2$:", RSquared)
slt.write(f"Coefficient: {popt[0]}\n\nPower:{popt[1]}\n\nConstant:{popt[2]}")
slt.pyplot(fig)

fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True, sharey=True)
Constants = [popt[2]]
for i in range(1, 20):
    tempDict = ThresholdAgainstTimeWidth
    tempDict[2 ** i] = 0.06
    timeWidth = np.array(list(tempDict.keys()))
    threshold = np.array(list(tempDict.values()))
    popt, P_predict = powerRegression(timeWidth, threshold)
    Constants.append(popt[2])
# fig = plt.figure(figsize=(7, 7))
axs[0].scatter([i for i in range(20)], Constants)
axs[0].plot(
    Constants, label="$threshold_{exp}$, given one extra point $threshold_{ob}=0.06$"
)
# axs[0].xlabel("log of stimulation duration (s) (in absolute unit)")
# axs[0].ylabel("Calculated threshold stimulus (V)")
axs[0].hlines(0.055, 0, len(Constants) - 1, color="tab:grey", alpha=0.5)
axs[0].fill_betweenx(
    [min(Constants), max(Constants) + 0.002], 9.5, 19, color="tab:grey", alpha=0.5
)
axs[0].legend()
# axis = plt.axis()
# axis
# slt.pyplot(fig)

Constants = [popt[2]]
for i in range(1, 20):
    tempDict = ThresholdAgainstTimeWidth
    tempDict[2 ** i] = 0.06
    tempDict[2 ** (i + 1)] = 0.06
    timeWidth = np.array(list(tempDict.keys()))
    threshold = np.array(list(tempDict.values()))
    popt, P_predict = powerRegression(timeWidth, threshold)
    Constants.append(popt[2])
# fig = plt.figure(figsize=(7, 7))
axs[1].scatter([i for i in range(20)], Constants)
axs[1].plot(
    Constants, label="$threshold_{exp}$, given two extra point $threshold_{ob}=0.06$"
)
# axs[1].xlim(axis[0], axis[1])
# axs[1].ylim(axis[2], axis[3])
axs[1].hlines(0.055, 0, len(Constants) - 1, color="tab:grey", alpha=0.5)
# plt.fill_betweenx([min(Constants), max(Constants) + 0.002],
#                   9.5,
#                   19,
#                   color="tab:grey",
#                   alpha=0.5)
axs[1].legend()
fig.text(0.5, 0.05, "log of stimulus duration (s) (in absolute unit)", ha="center")
fig.text(
    0.02, 0.5, "Calculated threshold stimulus (V)", va="center", rotation="vertical"
)
slt.pyplot(fig)


def normalDistributionLine(x: np.array, mu: float, sigma: float, Amplitude: float):
    dSquared = np.power((x - mu) / sigma, 2)
    return Amplitude / sigma / np.sqrt(2 * np.pi) * np.exp(-dSquared / 2)


x = np.linspace(0, 25, 2000000)
y1 = normalDistributionLine(x, 5, 1, 3)
y2 = normalDistributionLine(x, 8, 1.5, -3)
y3 = normalDistributionLine(x, 8, 1.5, -0.3)
# fig = plt.figure()
fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True, sharey=True)
axs[0].plot(x, y1, alpha=0.5, label="Positive phase")
axs[0].plot(x, y2, alpha=0.5, label="Negative phase")
axs[0].plot(x, y1 + y2, label="Biphasic action potential")
axs[0].legend()
axs[0].hlines(0, min(x), max(x), color="tab:grey", alpha=0.5)
axs[1].plot(x, y1, alpha=0.5, label="Positive phase")
axs[1].plot(x, y3, alpha=0.5, label="Reduced negative phase")
axs[1].plot(x, y1 + y3, label="Monophasic action potential")
axs[1].legend()
axs[1].hlines(0, min(x), max(x), color="tab:grey", alpha=0.5)
axs[2].plot(x, y1 + y2, label="Biphasic action potential")
axs[2].plot(x, y1 + y3, label="Monophasic action potential")
axs[2].legend()
axs[2].hlines(0, min(x), max(x), color="tab:grey", alpha=0.5)
fig.text(0.5, 0.05, "Time", ha="center")
fig.text(0.02, 0.5, "Simulated signal voltage", va="center", rotation="vertical")
slt.pyplot(fig)

