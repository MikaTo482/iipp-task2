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
