import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from utils_save_result import save_result

def save_figure(fig, script_name, filename, base_dir="graph", dpi=150):
    target_dir = os.path.join(base_dir, script_name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = os.path.join(target_dir, filename)
    
    if os.path.exists(target_path):
        os.remove(target_path)
    
    fig.savefig(target_path, dpi=dpi, bbox_inches="tight")

X = pd.read_csv("secom/secom.data", sep=r"\s+", header=None)

y = pd.read_csv("secom/secom_labels.data", sep=r"\s+", header=None,
                           names=["label", "timestamp"])

script_name = "02_temporal_audit.py"
graph1_name = "observation_count_over_time.png"
graph2_name = "fail_count_over_time.png"
graph3_name = "fail_rate_over_time.png"


y['timestamp'] = pd.to_datetime(y['timestamp'])
y['label'] = y['label'].map({-1: 0, 1: 1})
print("Earliest Timestamp:", y['timestamp'].min())
print("Latest Timestamp:", y['timestamp'].max())
print("Number of Unique Timestamps:", y['timestamp'].nunique())

y['date'] = y['timestamp'].dt.date

grouped_by_date = y.groupby('date').agg({'label': ['count', 'sum', 'mean']}).reset_index()
grouped_by_date['date'] = pd.to_datetime(grouped_by_date['date'])


# ---------------------------------------------------------------
# Temporal Cutoff Analysis: Determine the optimal temporal cutoff for splitting the dataset into training and testing sets.
# ---------------------------------------------------------------

def find_temporal_cutoff_candidates(y, target_fail_counts=[15, 20, 25]):
    """
    หา Temporal Cutoff Candidates โดยใช้ Timestamp + Fail Count เท่านั้น
    (ห้ามดู Model Performance มาช่วยตัดสินใจ)

    สำหรับแต่ละ target_fail_count จะหาจุดตัดที่ทำให้ Test period
    (ข้อมูลหลัง cutoff) มี Fail observation เท่ากับเป้าหมายพอดี
    โดยให้ Dev period (ก่อน cutoff) มีขนาดใหญ่ที่สุดเท่าที่จะเป็นไปได้

    Parameters
    ----------
    y : DataFrame ที่มี column 'label' (-1/+1) และ 'timestamp' (datetime)
    target_fail_counts : list ของจำนวน Fail เป้าหมายที่ต้องการใน Test period

    Returns
    -------
    list ของ dict แต่ละ candidate
    """
    # ขั้นที่ 1-2: เรียงตามเวลา
    df = y.sort_values("timestamp").reset_index(drop=True)
    df["is_fail"] = (df["label"] == 1).astype(int)

    # ขั้นที่ 3: นับจำนวน Fail ที่เหลือ "หลัง" แต่ละจุด (ไม่รวมตัวเอง)
    df["fails_strictly_after"] = df["is_fail"][::-1].cumsum()[::-1] - df["is_fail"]

    candidates = []
    seen_cutoffs = set()

    for target in sorted(set(target_fail_counts)):
        # ขั้นที่ 4: หาจุดตัดที่ตรงเป้าหมาย
        eligible = df[df["fails_strictly_after"] >= target]
        if eligible.empty:
            continue

        cutoff_idx = eligible.index[-1]
        cutoff_time = df.loc[cutoff_idx, "timestamp"]

        # ป้องกันคู่ซ้ำ (ถ้า target ต่างกันแต่ cutoff ตรงกันพอดี)
        if cutoff_time in seen_cutoffs:
            continue
        seen_cutoffs.add(cutoff_time)

        test_portion = df[df["timestamp"] > cutoff_time]
        dev_portion = df[df["timestamp"] <= cutoff_time]

        candidates.append({
            "target_fail_count": target,
            "cutoff_timestamp": cutoff_time,
            "dev_n": int(len(dev_portion)),
            "dev_fail_count": int(dev_portion["is_fail"].sum()),
            "test_n": int(len(test_portion)),
            "test_fail_count": int(test_portion["is_fail"].sum()),
            "test_fail_rate_pct": round(
                test_portion["is_fail"].sum() / len(test_portion) * 100, 3
            ) if len(test_portion) > 0 else None,
        })

    return candidates

candidates = find_temporal_cutoff_candidates(y, target_fail_counts=[15, 20, 25])


# ---------------------------------------------------------------
# Generate a summary report of the temporal audit results and save it to a text file.
# ---------------------------------------------------------------

content = f"""
Earliest Timestamp: {y['timestamp'].min()}\n
Latest Timestamp: {y['timestamp'].max()}\n
Number of Unique Timestamps: {y['timestamp'].nunique()}\n
==========================\n
Temporal Cutoff Candidates:\n
==========================\n
"""

for i, c in enumerate(candidates, 1):
    content += f"\nCandidate {i}:\n"
    for k, v in c.items():
        content += f"    {k}: {v}\n"

save_result(
    content=content,
    script_name=script_name,
    filename="02_report.txt"
)

# ---------------------------------------------------------------
# SECOM: Observation Count over Time (by day)
# ---------------------------------------------------------------

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#333333",
    "figure.facecolor": "white",
})

fig1, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(grouped_by_date['date'], grouped_by_date['label']['count'],
        color="#BE280A", linewidth=1, marker="o", markersize=3)

ax.set_title("SECOM: Observation Count over Time (by day)", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Observations")
ax.set_xlabel("")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(axis="y", alpha=0.25)

for label in ax.get_xticklabels():
    label.set_rotation(40)
    label.set_ha("right")

fig1.tight_layout()

save_figure(fig1, script_name=script_name, filename=graph1_name)
plt.close(fig1)
print(f"Saved -> graph/{script_name}/{graph1_name}")

# ---------------------------------------------------------------
# SECOM: Fail Count over Time (by day)
# ---------------------------------------------------------------

fig2, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(grouped_by_date['date'], grouped_by_date['label']['sum'],
        color="#BE280A", linewidth=1, marker="o", markersize=3)

ax.set_title("SECOM: Fail Count over Time (by day)", fontsize=13, fontweight="bold")
ax.set_ylabel("Number of Failures")
ax.set_xlabel("")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(axis="y", alpha=0.25)

for label in ax.get_xticklabels():
    label.set_rotation(40)
    label.set_ha("right")

fig2.tight_layout()

save_figure(fig2, script_name=script_name, filename=graph2_name)
plt.close(fig2)
print(f"Saved -> graph/{script_name}/{graph2_name}")

# ---------------------------------------------------------------
# SECOM: Fail Rate over Time (by day)
# ---------------------------------------------------------------

fig3, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(grouped_by_date['date'], grouped_by_date['label']['mean'],
        color="#BE280A", linewidth=1, marker="o", markersize=3)

ax.set_title("SECOM: Fail Rate over Time (by day)", fontsize=13, fontweight="bold")
ax.set_ylabel("Fail Rate")
ax.set_xlabel("")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(axis="y", alpha=0.25)

for label in ax.get_xticklabels():
    label.set_rotation(40)
    label.set_ha("right")

fig3.tight_layout()

save_figure(fig3, script_name=script_name, filename=graph3_name)
plt.close(fig3)
print(f"Saved -> graph/{script_name}/{graph3_name}")
