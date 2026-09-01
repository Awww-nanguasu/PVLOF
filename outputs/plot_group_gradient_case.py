# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

input_path = Path(
    r"outputs\gradient_under_current_234_6T2549000365_20260703_1145_full.csv"
)

# Python内部先使用纯英文路径，避免Windows管道编码问题
output_path = Path(
    r"outputs\group_gradient_undercurrent_case.png"
)

df = pd.read_csv(
    input_path,
    encoding="utf-8-sig",
    parse_dates=["event_time_local"],
).set_index("event_time_local")

df.columns = [
    str(column).zfill(2)
    for column in df.columns
]

low_strings = {
    "11", "12", "13", "14", "16", "20"
}

normal_colors = {
    "01": "#4DB6AC",
    "02": "#4C78A8",
    "03": "#8064A2",
    "04": "#C47AA0",
    "05": "#E66C5C",
    "10": "#D18F43",
    "15": "#7AA6C2",
    "19": "#B59A6A",
    "21": "#5B8FF9",
}

low_colors = {
    "11": "#E76F8A",
    "12": "#D94F70",
    "13": "#C83E65",
    "14": "#F08AA0",
    "16": "#B52F58",
    "20": "#FF6685",
}

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(
    figsize=(16, 8),
    facecolor="white",
)

fig.subplots_adjust(
    left=0.07,
    right=0.79,
    bottom=0.09,
    top=0.87,
)

for string_no in df.columns:
    if string_no in low_strings:
        color = low_colors[string_no]
        linewidth = 2.3
        alpha = 1.0
        zorder = 4
    else:
        color = normal_colors.get(
            string_no,
            "#8093A8",
        )
        linewidth = 1.45
        alpha = 0.82
        zorder = 2

    ax.plot(
        df.index,
        df[string_no],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=f"string_current_{string_no} 的平均值",
        zorder=zorder,
    )

ax.set_title(
    "群体梯度欠流实测案例",
    fontsize=20,
    fontweight="bold",
    color="#15264B",
    pad=22,
)

ax.text(
    0.5,
    1.015,
    "电站234｜设备6T2549000365｜2026-07-03 11:45",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=11,
    color="#52627A",
)

ax.set_ylabel(
    "组串电流（A）",
    fontsize=11,
    fontweight="bold",
    color="#38475A",
)

ax.set_xlabel(
    "时间（5分钟）",
    fontsize=10,
    fontweight="bold",
    color="#38475A",
    labelpad=12,
)

ax.set_ylim(0, 18)

ax.set_xticks(df.index)
ax.set_xticklabels(
    [
        timestamp.strftime("%H:%M")
        for timestamp in df.index
    ],
    fontsize=9,
)

ax.tick_params(
    axis="y",
    labelsize=9,
    colors="#59687A",
)

ax.tick_params(
    axis="x",
    colors="#59687A",
)

ax.grid(
    axis="both",
    color="#DCE5EF",
    linestyle="--",
    linewidth=0.8,
    alpha=0.85,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#B8C5D4")
ax.spines["bottom"].set_color("#B8C5D4")

legend = ax.legend(
    loc="center left",
    bbox_to_anchor=(1.01, 0.5),
    frameon=False,
    fontsize=8.5,
    handlelength=2.2,
    labelspacing=0.75,
)

for legend_text in legend.get_texts():
    string_no = (
        legend_text
        .get_text()
        .split("_")[2]
        .split()[0]
    )

    if string_no in low_strings:
        legend_text.set_color("#B52F58")
        legend_text.set_fontweight("bold")
    else:
        legend_text.set_color("#52627A")

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

fig.savefig(
    output_path,
    dpi=220,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)

print("OUTPUT:", output_path.resolve())
