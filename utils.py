"""
utils_save_result.py
=====================
Reusable helper to save text (.txt) result files into an organized
folder structure: result/<script_name>/<filename>.txt

Mirrors the same behavior as save_figure():
    - If the target folder does not exist -> create it (including parents).
    - If a file with the same name already exists -> delete it first,
      then save the new version (no stale leftovers from a previous run).
"""

import os


def save_result(content: str, script_name: str, filename: str, base_dir: str = "result"):
    """
    Save a text string to: {base_dir}/{script_name}/{filename}

    Parameters
    ----------
    content : str
        The text content to write into the file.
    script_name : str
        Subfolder name, typically the script that generated the result
        (e.g. "01_secom_data_audit").
    filename : str
        Output file name, e.g. "class_distribution.txt".
    base_dir : str
        Root folder for all results (default "result").
    """
    target_dir = os.path.join(base_dir, script_name)

    # 1. Create folder (and parents) if it doesn't exist yet
    os.makedirs(target_dir, exist_ok=True)

    target_path = os.path.join(target_dir, filename)

    # 2. If the file already exists, delete it before saving a fresh copy
    if os.path.exists(target_path):
        os.remove(target_path)
        print(f"Existing file removed: {target_path}")

    # 3. Save the new content
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Result saved -> {target_path}")
    return target_path


def find_temporal_cutoff_candidates(cutoff_input, target_fail_counts=[15, 20, 25]):
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
    dict
        key = target_fail_count
        value = dict {
            "cutoff_timestamp", "dev_n", "dev_fail_count",
            "test_n", "test_fail_count", "test_fail_rate_pct"
        }
    """
    # ขั้นที่ 1-2: เรียงตามเวลา
    cutoff_input = cutoff_input.sort_values("timestamp").reset_index(drop=True)
    cutoff_input["is_fail"] = (cutoff_input["label"] == 1).astype(int)

    # ขั้นที่ 3: นับจำนวน Fail ที่เหลือ "หลัง" แต่ละจุด (ไม่รวมตัวเอง)
    cutoff_input["fails_strictly_after"] = cutoff_input["is_fail"][::-1].cumsum()[::-1] - cutoff_input["is_fail"]

    candidates = {}
    seen_cutoffs = set()

    for target in sorted(set(target_fail_counts)):
        # ขั้นที่ 4: หาจุดตัดที่ตรงเป้าหมาย
        eligible = cutoff_input[cutoff_input["fails_strictly_after"] >= target]
        if eligible.empty:
            continue

        cutoff_idx = eligible.index[-1]
        cutoff_time = cutoff_input.loc[cutoff_idx, "timestamp"]

        # ป้องกันคู่ซ้ำ (ถ้า target ต่างกันแต่ cutoff ตรงกันพอดี)
        if cutoff_time in seen_cutoffs:
            continue
        seen_cutoffs.add(cutoff_time)

        test_portion = cutoff_input[cutoff_input["timestamp"] > cutoff_time]
        dev_portion = cutoff_input[cutoff_input["timestamp"] <= cutoff_time]

        test_fail_rate_pct = round(
            test_portion["is_fail"].sum() / len(test_portion) * 100, 3
        ) if len(test_portion) > 0 else None

        candidates[target] = {
            "cutoff_timestamp": cutoff_time,
            "dev_n": int(dev_portion.shape[0]),
            "dev_fail_count": int(dev_portion["is_fail"].sum()),
            "test_n": int(test_portion.shape[0]),
            "test_fail_count": int(test_portion["is_fail"].sum()),
            "test_fail_rate_pct": test_fail_rate_pct,
        }

    return candidates
