import streamlit as slt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition, mark_inset


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


eyeToPaperDist = 70
eyeballDiameter = 1.5

rightEyeOnPaper = np.array(
    [
        [-3.75, -0.20],
        [-2.10, 2.90],
        [0.95, 4.00],
        [3.75, 2.75],
        [4.65, -0.30],
        [4.85, -3.80],
        [1.30, -4.85],
        [-2.10, -3.70],
    ]
)
leftEyeOnPaper = np.array(
    [
        [-2.80, 0.05],
        [-2.50, 3.70],
        [0.30, 6.60],
        [2.60, 5.60],
        [5.20, 3.30],
        [5.30, 0.35],
        [3.70, -1.05],
        [0.95, -3.80],
        [-1.85, -2.10],
    ]
)
badDataPoint_left = np.array([[5.50, 4.90]])

rightEyeOnPaper[:, 0] += 20
leftEyeOnPaper[:, 0] += 20
leftEyeOnPaper *= -1
badDataPoint_left[:, 0] += 20
badDataPoint_left *= -1
rightEye = rightEyeOnPaper * -eyeballDiameter / eyeToPaperDist
leftEye = leftEyeOnPaper * -eyeballDiameter / eyeToPaperDist

# "rightEyeOnPaper", rightEyeOnPaper
# "leftEyeOnPaper", leftEyeOnPaper
# "badDataPoint_left", badDataPoint_left
# "rightEye", rightEye
# "leftEye", leftEye

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
ax.vlines(x=0, ymax=10, ymin=-10, label=None, color="tab:grey", alpha=0.1)
ax.hlines(y=0, xmax=30, xmin=-30, label=None, color="tab:grey", alpha=0.1)
ax.scatter(
    rightEyeOnPaper[:, 0],
    rightEyeOnPaper[:, 1],
    color="tab:blue",
    label=None,
)
ax.fill(
    rightEyeOnPaper[:, 0],
    rightEyeOnPaper[:, 1],
    color="tab:blue",
    alpha=0.1,
    label="Right eye blind spot on paper",
)
ax.fill(
    rightEye[:, 0],
    rightEye[:, 1],
    color="tab:blue",
    alpha=0.5,
    label="Right eye blind spot (approximate)",
)
ax.scatter(
    leftEyeOnPaper[:, 0],
    leftEyeOnPaper[:, 1],
    color="tab:orange",
    label=None,
)
ax.fill(
    leftEyeOnPaper[:, 0],
    leftEyeOnPaper[:, 1],
    color="tab:orange",
    alpha=0.1,
    label="Left eye blind spot on paper",
)
ax.fill(
    leftEye[:, 0],
    leftEye[:, 1],
    color="tab:orange",
    alpha=0.5,
    label="Left eye blind spot (approximate)",
)
ax.scatter(
    badDataPoint_left[:, 0],
    badDataPoint_left[:, 1],
    marker="x",
    color="tab:red",
    alpha=0.5,
    label="Outlier",
)
ax.scatter(
    0, 0, marker="^", color="tab:grey", label="Eye focus (after mapping)", alpha=0.5
)
ax.set_xlabel("Relative $X$ position to eye focus")
ax.set_ylabel("Relative $Y$ position to eye focus")
ax.legend(
    bbox_to_anchor=(0, 1.02, 1, 0.102),
    loc="lower left",
    ncol=3,
    mode="expand",
    borderaxespad=0.0,
)
ax.set_xlim(-30, 30)
ax.set_ylim(-10, 10)

subpos = [0.4, 0.1, 0.2, 0.2]
subax = plt.axes([-3, -1, 3, 1])
ip = InsetPosition(ax, subpos)
subax.set_axes_locator(ip)
mark_inset(ax, subax, loc1=1, loc2=2, fc="none", ec="0.5")
subax.set_xlim(-0.9, 0.9)
subax.set_ylim(-0.3, 0.3)

subax.vlines(x=0, ymax=10, ymin=-10, label=None, color="tab:grey", alpha=0.1)
subax.hlines(y=0, xmax=30, xmin=-30, label=None, color="tab:grey", alpha=0.1)
subax.fill(
    rightEye[:, 0],
    rightEye[:, 1],
    color="tab:blue",
    alpha=0.5,
    label=f"$S_R={PolyArea(rightEye[:,0],rightEye[:,1])*100:>.2f}\ mm^2$",
)
subax.fill(
    leftEye[:, 0],
    leftEye[:, 1],
    color="tab:orange",
    alpha=0.5,
    label=f"$S_L={PolyArea(leftEye[:,0],leftEye[:,1])*100:>.2f}\ mm^2$",
)
subax.scatter(0, 0, marker="^", color="tab:grey", label=None, alpha=0.5)
subax.legend(
    bbox_to_anchor=(0, 1.12, 1, 0.102),
    loc="center",
    ncol=2,
    mode="center",
    borderaxespad=0,
    fontsize=8,
)

slt.pyplot(fig)

concentrations = [0.1, 0.2, 0.5, 1]

frogLeft = np.array(
    [
        [0.1, 1 / 24.13],
        [0.1, 1 / 8.52],
        [0.1, 1 / 13.18],
        [0.1, 1 / 22.83],
        [0.1, 0],
        [0.1, 0],
        [0.2, 0],
        [0.2, 0],
        [0.2, 0],
        [0.2, 1 / 21.88],
        [0.2, 1 / 9.49],
        [0.2, 1 / 17.38],
        [0.2, 1 / 13.49],
        [0.5, 1 / 5.45],
        [0.5, 1 / 5.86],
        [0.5, 1 / 8.55],
        [0.5, 1 / 10.52],
        [0.5, 1 / 2.81],
        [0.5, 1 / 3.53],
        [1.0, 1 / 9.16],
        [1.0, 1 / 4.59],
        [1.0, 1 / 5.60],
        [1.0, 1 / 5.64],
        [1.0, 1 / 2.39],
        [1.0, 1 / 5.14],
    ]
)
frogRight = np.array(
    [
        [0.1, 1 / 11.00],
        [0.1, 1 / 7.69],
        [0.1, 1 / 10.17],
        [0.1, 1 / 9.30],
        [0.1, 0],
        [0.1, 0],
        [0.2, 1 / 8.33],
        [0.2, 1 / 14.57],
        [0.2, 1 / 4.95],
        [0.2, 1 / 9.81],
        [0.2, 1 / 9.13],
        [0.5, 1 / 4.11],
        [0.5, 1 / 7.78],
        [0.5, 1 / 1.90],
        [0.5, 1 / 6.95],
        [0.5, 1 / 6.41],
        [1.0, 1 / 1.79],
        [1.0, 1 / 1.83],
        [1.0, 1 / 2.66],
        [1.0, 1 / 3.74],
        [1.0, 1 / 6.64],
    ]
)

dataLeft = []
dataRight = []

for i in range(len(concentrations)):
    dataLeft.append(frogLeft[frogLeft[:, 0] == concentrations[i], 1])
    dataRight.append(frogRight[frogRight[:, 0] == concentrations[i], 1])

# fig = plt.figure()
fig, axes = plt.subplots(figsize=(10, 5), nrows=1, ncols=2, sharex=True, sharey=True)
plt.tight_layout(pad=0)
# ax = fig.add_subplot(111)
axes[0].scatter(
    # frogLeft[:, 0] + 0.01 * np.random.randn(len(frogLeft)),
    frogLeft[:, 0],
    frogLeft[:, 1],
    color="tab:blue",
    alpha=0.5,
    label="Left frog's left leg",
)
axes[1].scatter(
    # frogRight[:, 0] + 0.01 * np.random.randn(len(frogRight)),
    frogRight[:, 0],
    frogRight[:, 1],
    color="tab:green",
    alpha=0.5,
    label="Right frog's left leg",
)

values = axes[0].boxplot(dataLeft, positions=concentrations, widths=0.085)
values = axes[1].boxplot(dataRight, positions=concentrations, widths=0.085)
plt.xlim(0, 1.1)
axes[0].legend()
axes[1].legend()
axes[0].set_xlabel("$c_{H_2SO_4}$ (%)")
axes[1].set_xlabel("$c_{H_2SO_4}$ (%)")
axes[0].set_ylabel(r"$\frac{1}{\tau_{response}}$ ($s^{-1}$)")
# plt.boxplot(data,)
slt.pyplot(fig)

[np.mean(grp) for grp in dataLeft]

top = np.array([0, 0, 70])

leftData = np.concatenate((leftEyeOnPaper, np.zeros((len(leftEyeOnPaper), 1))), axis=1)
rightData = np.concatenate(
    (rightEyeOnPaper, np.zeros((len(rightEyeOnPaper), 1))), axis=1
)


def calcOmega(top, px, py, pz):
    A = px - top
    B = py - top
    C = pz - top
    alpha = np.arccos(np.dot(B, C) / (np.sqrt(B.dot(B.T)) * np.sqrt(C.dot(C.T))))
    beta = np.arccos(np.dot(A, C) / (np.sqrt(A.dot(A.T)) * np.sqrt(C.dot(C.T))))
    gamma = np.arccos(np.dot(A, B) / (np.sqrt(A.dot(A.T)) * np.sqrt(B.dot(B.T))))
    s = (alpha + beta + gamma) / 2
    # alpha, beta, gamma, s
    return 4 * np.arctan(
        np.sqrt(
            np.tan(s / 2)
            * np.tan((s - alpha) / 2)
            * np.tan((s - beta) / 2)
            * np.tan((s - gamma) / 2)
        )
    )


leftRoot = np.mean(leftData, axis=0)
leftRoot
rightRoot = np.mean(rightData, axis=0)
rightRoot

leftOmega = 0

for idx in range(0, len(leftData) - 1):
    leftOmega += calcOmega(top, leftRoot, leftData[idx, :], leftData[idx + 1, :])

leftOmega * (1.5 ** 2)

PolyArea(leftEye[:, 0], leftEye[:, 1]) * 1

rightOmega = 0

for idx in range(1, len(rightData) - 1):
    rightOmega += calcOmega(top, rightRoot, rightData[idx, :], rightData[idx + 1, :])

rightOmega * (1.5 ** 2)

PolyArea(rightEye[:, 0], rightEye[:, 1]) * 1