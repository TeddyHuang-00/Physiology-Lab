from numpy import median, var, mean
import numpy as np
import streamlit as slt
import os
import pandas as pd
import matplotlib.pyplot as plt

fileList = [fileName for fileName in os.listdir(".") if fileName[-3:] == "xls"]
fileList = fileList[:-1]
# fileList

dfs = []
reactTime = []
for fileName in fileList:
    dfs.append(pd.read_excel(fileName))
    reactTime.append(dfs[-1].loc[dfs[-1]["正确与否"] == "正确", "反应时间(s)"].to_list())
# reactTime

ExpData = pd.DataFrame(
    {
        "TimeMinute": [0, 5, 31, 55, 80, 109, 135, 160],
        "AlcoholBloodConcentration": [0, 40.6, 70.4, 68.0, 55.4, 42.4, 47.5, 42.8],
        "HeartRate": [60, 62, 70, 102, 96, 79, 77, 72],
        "ImmediateMemoryPct": [
            13 / 15 * 100,
            13 / 15 * 100,
            15 / 15 * 100,
            12 / 15 * 100,
            12 / 15 * 100,
            15 / 15 * 100,
            12 / 15 * 100,
            14 / 15 * 100,
        ],
        "ShortTermMemoryPct": [
            13 / 15 * 100,
            13 / 15 * 100,
            10 / 15 * 100,
            14 / 15 * 100,
            13 / 15 * 100,
            13 / 15 * 100,
            14 / 15 * 100,
            14 / 15 * 100,
        ],
        "ArithmeticTotalTime": [
            65.195,
            77.133,
            78.999,
            72.714,
            67.977,
            86.515,
            78.234,
            70.931,
        ],
        "ArithmeticCorrectnessPct": [
            15 / 15 * 100,
            14 / 15 * 100,
            14 / 15 * 100,
            14 / 15 * 100,
            15 / 15 * 100,
            14 / 15 * 100,
            14 / 15 * 100,
            15 / 15 * 100,
        ],
        "LogicalTime": [227, 161, 185, 164, 238, 264, 156, 248],
        "InterpretationCorrectnessPct": [
            12 / 15 * 100,
            10 / 15 * 100,
            5 / 15 * 100,
            11 / 15 * 100,
            12 / 15 * 100,
            12 / 15 * 100,
            11 / 15 * 100,
            10 / 15 * 100,
        ],
        "MedianReaction": [median(react) for react in reactTime],
        "MeanReaction": [mean(react) for react in reactTime],
        "VarReaction": [var(react) for react in reactTime],
        "ReactionCorrectnessPct": [len(react) / 20 * 100 for react in reactTime],
    }
)
ExpData = pd.DataFrame(
    {
        "TimeMinute": [0, 15, 30, 60, 90, 120, 150],
        "AlcoholBloodConcentration": [
            0,
            77.6,
            119.2,
            128.9,
            115,
            110.9,
            120.6,
        ],
        "HeartRate": [
            89,
            99,
            105,
            105,
            98,
            94,
            114,
        ],
        "ImmediateMemoryPct": [
            9 / 15 * 100,
            5 / 15 * 100,
            4 / 15 * 100,
            8 / 15 * 100,
            10 / 15 * 100,
            10 / 15 * 100,
            7 / 15 * 100,
        ],
        "ShortTermMemoryPct": [
            8 / 15 * 100,
            8 / 15 * 100,
            9 / 15 * 100,
            6 / 15 * 100,
            7 / 15 * 100,
            9 / 15 * 100,
            9 / 15 * 100,
        ],
        "ArithmeticTotalTime": [
            85,
            113,
            106,
            81,
            98,
            85,
            93,
        ],
        "ArithmeticCorrectnessPct": [
            13 / 15 * 100,
            13 / 15 * 100,
            10 / 15 * 100,
            12 / 15 * 100,
            12 / 15 * 100,
            14 / 15 * 100,
            12 / 15 * 100,
        ],
        "LogicalTime": [
            323,
            354,
            279,
            347,
            306,
            357,
            271,
        ],
        "InterpretationCorrectnessPct": [
            12 / 15 * 100,
            9 / 15 * 100,
            10 / 15 * 100,
            8 / 15 * 100,
            4 / 15 * 100,
            12 / 15 * 100,
            11 / 15 * 100,
        ],
        "MeanReaction": [
            415 / 1000,
            476 / 1000,
            504 / 1000,
            634 / 1000,
            467 / 1000,
            519 / 1000,
            447 / 1000,
        ],
        "ReactionCorrectnessPct": [
            93,
            100,
            87,
            93,
            80,
            87,
            87,
        ],
    }
)
ExpData
# slt.table(ExpData)
# ExpData.to_csv("ExperimentData.csv")

# fig = plt.figure(figsize=(7, 7))
fig, (axU, axD) = plt.subplots(nrows=2, ncols=1, sharex=True)
axU.plot(
    ExpData.TimeMinute,
    ExpData.AlcoholBloodConcentration,
    label=r"Alcohol Concentration / $mg\cdot dL^{-1}$",
)
axU.plot(ExpData.TimeMinute, ExpData.HeartRate, label=r"Heart Rate / BPM")
axU.plot(
    ExpData.TimeMinute,
    ExpData.ImmediateMemoryPct,
    # label=r"Immediate Memory (correctness, $\tau_1$=1.5s) / %",
    label=r"Immediate Memory (correctness) / %",
)
axU.plot(
    ExpData.TimeMinute,
    ExpData.ShortTermMemoryPct,
    # label=r"Short Term Memory (correctness, $\tau_2$=10s) / %",
    label=r"Short Term Memory (correctness) / %",
)
axU.plot(
    ExpData.TimeMinute,
    ExpData.ArithmeticCorrectnessPct,
    label=r"Arithmetic (correctness) / %",
)
axU.plot(
    ExpData.TimeMinute,
    ExpData.ArithmeticTotalTime,
    label=r"Arithmetic (total time, $N$=15) / s",
)
axU.plot(ExpData.TimeMinute, ExpData.LogicalTime, label=r"Logical / s")
axU.plot(
    ExpData.TimeMinute,
    ExpData.InterpretationCorrectnessPct,
    label=r"Language Interpretation (correctness) / %",
)
axU.plot(
    ExpData.TimeMinute,
    ExpData.MeanReaction * 1000,
    label=r"Rection (average time) / ms",
)
axU.plot(
    ExpData.TimeMinute,
    ExpData.ReactionCorrectnessPct,
    label=r"Reaction (correctness) / %",
)
axU.legend(
    bbox_to_anchor=(0, 1.25, 1, 0.102),
    loc="center",
    ncol=2,
    mode="center",
    borderaxespad=0,
    fontsize=8,
)
axD.plot(
    ExpData.TimeMinute,
    np.log2(ExpData.AlcoholBloodConcentration / ExpData.AlcoholBloodConcentration[0]),
)
axD.plot(ExpData.TimeMinute, np.log2(ExpData.HeartRate / ExpData.HeartRate[0]))
axD.plot(
    ExpData.TimeMinute,
    np.log2(ExpData.ImmediateMemoryPct / ExpData.ImmediateMemoryPct[0]),
)
axD.plot(
    ExpData.TimeMinute,
    np.log2(ExpData.ShortTermMemoryPct / ExpData.ShortTermMemoryPct[0]),
)
axD.plot(
    ExpData.TimeMinute,
    np.log2(ExpData.ArithmeticCorrectnessPct / ExpData.ArithmeticCorrectnessPct[0]),
)
axD.plot(
    ExpData.TimeMinute,
    np.log2(ExpData.ArithmeticTotalTime / ExpData.ArithmeticTotalTime[0]),
)
axD.plot(ExpData.TimeMinute, np.log2(ExpData.LogicalTime / ExpData.LogicalTime[0]))
axD.plot(
    ExpData.TimeMinute,
    np.log2(
        ExpData.InterpretationCorrectnessPct / ExpData.InterpretationCorrectnessPct[0]
    ),
)
axD.plot(
    ExpData.TimeMinute,
    np.log2(ExpData.MeanReaction / ExpData.MeanReaction[0]),
)
axD.plot(
    ExpData.TimeMinute,
    np.log2(ExpData.ReactionCorrectnessPct / ExpData.ReactionCorrectnessPct[0]),
)
axD.set_xlabel(r"Time / min")
axU.set_ylabel(r"Absolute unit")
axD.set_ylabel(r"$\log_2$-fold change")
axU.set_title("Test results from experiment", pad=75)
slt.pyplot(fig)

ExpData.columns

selectedColumns = ["HeartRate", "LogicalTime", "InterpretationCorrectnessPct"]

VarExpData = ExpData.iloc[:, :]

for i in range(2, len(VarExpData.columns)):
    VarExpData.iloc[:, i] = np.log2(VarExpData.iloc[:, i] / VarExpData.iloc[0, i])

slt.dataframe(VarExpData)

from sklearn.decomposition import PCA

pc = PCA(n_components=2)
pc_res = pc.fit(VarExpData.iloc[:, 2:])
pc_coord = pc.fit_transform(VarExpData.iloc[:, 2:])

fig = plt.figure(figsize=(9, 7))
plt.quiver(
    pc_coord[:-1, 0],
    pc_coord[:-1, 1],
    pc_coord[1:, 0] - pc_coord[:-1, 0],
    pc_coord[1:, 1] - pc_coord[:-1, 1],
    scale_units="xy",
    angles="xy",
    scale=1,
    color="tab:gray",
    # label=VarExpData.columns[2:],
)
plt.quiver(
    np.zeros_like(pc.components_[0, :])
    + (max(pc_coord[:, 0]) + min(pc_coord[:, 0])) / 2,
    np.zeros_like(pc.components_[1, :])
    + (max(pc_coord[:, 1]) + min(pc_coord[:, 1])) / 2,
    pc.components_[0, :],
    pc.components_[1, :],
    scale_units="xy",
    angles="xy",
    scale=1,
    width=0.002,
    color="tab:gray",
)
for i in range(len(pc.components_[0, :])):
    plt.annotate(
        VarExpData.columns[2 + i],
        (
            (max(pc_coord[:, 0]) + min(pc_coord[:, 0])) / 2
            + pc.components_[0, i] / 1,
            (max(pc_coord[:, 1]) + min(pc_coord[:, 1])) / 2
            + pc.components_[1, i] / 1,
        ),
    )
plt.scatter(
    pc_coord[:, 0],
    pc_coord[:, 1],
    s=ExpData.TimeMinute + 10,
    c=ExpData.AlcoholBloodConcentration,
    cmap="coolwarm",
)
for i in range(len(pc_coord[:, 0])):
    plt.annotate(ExpData.AlcoholBloodConcentration[i], (pc_coord[i, 0], pc_coord[i, 1]))
# plt.legend()
# plt.grid()
plt.colorbar()
plt.title("PCA of test results")
plt.xlabel("PC1")
plt.ylabel("PC2")
slt.pyplot(fig)

pc.components_
