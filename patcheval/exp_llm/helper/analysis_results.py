# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from collections import defaultdict
from textwrap import fill
from typing import Any, Dict, List
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from .logger import get_logger

# Language mapping
LANGUAGES = {
    "Python": ["py", "Py", "python", "Python"],
    "Go": ["go", "Go"],
    "JavaScript": ["js", "JS", "JavaScript", "javascript", "npm", "TypeScript"],
}
LANG_TO_LANGUAGE: Dict[str, str] = {}
for category, langs in LANGUAGES.items():
    for lang in langs:
        LANG_TO_LANGUAGE[lang] = category

def read_json(file_path: str) -> Any:
    logger = get_logger()
    """Read a JSON file with UTF-8 encoding."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        raise

def extract_cve_info(data_infos: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """Extract CVE basic info like CWE id and language category."""
    cve_info: Dict[str, Dict[str, str]] = {}
    for info in data_infos:
        cve = info.get("cve_id", "")
        if not cve:
            continue
        cwe_id = info.get("cwe_id", [""])[0]
        language = info.get("language", "")
        cve_info[cve] = {
            "cwe_id": cwe_id,
            "language": LANG_TO_LANGUAGE.get(language, language),
        }
    return cve_info


def process_result_data(
    result_datas: List[Dict[str, List[Dict[str, Any]]]],
    cve_info: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Flatten result JSONs into a tabular list of rows per CVE including epoch columns."""
    all_epochs = set()
    for result_data in result_datas:
        for items in result_data.values():
            for item in items:
                all_epochs.add(item.get("epoch", 1))
    max_epoch = max(all_epochs) if all_epochs else 1

    table_data: List[Dict[str, Any]] = []
    processed_cves = set()

    for result_data in result_datas:
        for cve, items in result_data.items():
            if cve not in cve_info or cve in processed_cves:
                continue
            processed_cves.add(cve)

            row: Dict[str, Any] = {
                "cve": cve,
                "cwe": cve_info[cve]["cwe_id"],
                "url": "",
                "groundtruth": "",
                "language": cve_info[cve]["language"],
                "test_msg": "",
                "diff_content": "",
            }

            epoch_data: Dict[int, Dict[str, Any]] = {}
            for item in items:
                epoch = item.get("epoch", 1)
                status = item.get("status", "")
                if status is None:
                    continue

                # Normalize fix_code to a string (support dict and list formats)
                fix_code = ""
                if item.get("fix_code"):
                    if isinstance(item["fix_code"], dict):
                        parts = []
                        for vul_id, code in item["fix_code"].items():
                            parts.append(f"/* {vul_id} */\n" + code)
                        fix_code = "\n\n".join(parts) if parts else ""
                    else:
                        fix_code = "\n".join([code for code in item["fix_code"] if code.strip()])

                if not row["url"] and item.get("url"):
                    row["url"] = "\n".join(item.get("url", []))
                if not row["groundtruth"] and item.get("groundtruth"):
                    row["groundtruth"] = item.get("groundtruth", "")
                if not row["test_msg"] and item.get("test_msg"):
                    row["test_msg"] = item.get("test_msg", "")
                if not row["diff_content"] and item.get("diff_content"):
                    row["diff_content"] = item.get("diff_content", "")

                epoch_data[epoch] = {"fix_code": fix_code, "status": status}

            for epoch in range(1, max_epoch + 1):
                if epoch in epoch_data:
                    row[f"epoch{epoch}_fix_code"] = epoch_data[epoch]["fix_code"]
                    row[f"epoch{epoch}_status"] = epoch_data[epoch]["status"]
                else:
                    row[f"epoch{epoch}_fix_code"] = ""
                    row[f"epoch{epoch}_status"] = ""

            table_data.append(row)

    return table_data


def generate_statistics(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generate simple per-epoch success/fail statistics."""
    if df.empty:
        return {"by_epoch": pd.DataFrame()}
    
    status_cols = [col for col in df.columns if col.startswith("epoch") and "_status" in col]
    if not status_cols:
        return {"by_epoch": pd.DataFrame()}

    epochs = sorted({int(col.split("_")[0][5:]) for col in status_cols})
    epoch_stats = []
    for epoch in epochs:
        status_col = f"epoch{epoch}_status"
        if status_col not in df.columns:
            continue
        counts = df[status_col].value_counts().to_dict()
        epoch_stats.append(
            {
                "epoch": epoch,
                "success": counts.get("success", 0),
                "fail": counts.get("fail", 0),
                "other": len(df) - counts.get("success", 0) - counts.get("fail", 0),
                "total": len(df),
            }
        )

    return {"by_epoch": pd.DataFrame(epoch_stats).set_index("epoch")}


def adjust_column_width(text: Any, width: int = 10) -> str:
    """Wrap cell text to given width for better table display."""
    return fill(str(text), width=width)


def print_statistics(stats: Dict[str, pd.DataFrame], logger) -> None:
    """Print statistics tables using tabulate."""

    def prepare_dataframe(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.reset_index()
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: adjust_column_width(x, width=len(col) + 2))
        return df[columns]

    if not stats["by_epoch"].empty:
        epoch_columns = ["epoch", "success", "fail", "other", "total"]
        df_epoch = prepare_dataframe(stats["by_epoch"], epoch_columns)
        logger.info(
            "\n"
            + tabulate(
                df_epoch,
                headers=[adjust_column_width(h, width=len(h) + 2) for h in epoch_columns],
                tablefmt="grid",
                showindex=False,
            )
        )


def process_results_and_save(result_json_path: str, input_json_path: str) -> None:
    """Orchestrate reading, flattening, and summarizing result data."""
    logger = get_logger()

    # Step 1: Load JSONs
    data_infos = read_json(input_json_path)
    result_datas = read_json(result_json_path)

    # Step 2: Build dataframe
    cve_info = extract_cve_info(data_infos)
    df = pd.DataFrame(process_result_data(result_datas, cve_info))

    # Step 3: Generate and print stats
    stats = generate_statistics(df)
    print_statistics(stats, logger)


def merge_results(existing: List[Any], new: List[Any], flag: str | None = None) -> List[Any]:
    logger = get_logger()    
    if flag == "result":
        existing_dict: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))

        for item in existing:
            if not isinstance(item, dict):
                logger.warning(f"Skip invalid existing result (not dict): {item}")
                continue
            for cve_id, patch_list in item.items():
                if not isinstance(patch_list, list):
                    logger.warning(f"Skip invalid patch list for CVE {cve_id} (not list): {patch_list}")
                    continue
                for patch in patch_list:
                    if not isinstance(patch, dict) or "id" not in patch or "epoch" not in patch:
                        logger.warning(f"Skip invalid patch for CVE {cve_id} (missing id/epoch): {patch}")
                        continue
                    patch_id = patch["id"]
                    epoch = patch["epoch"]
                    existing_dict[cve_id][patch_id][epoch] = patch

        for entry in new:
            if not isinstance(entry, dict):
                logger.warning(f"Skip invalid new result (not dict): {entry}")
                continue
            for cve_id, new_patch_list in entry.items():
                if not isinstance(new_patch_list, list):
                    logger.warning(f"Skip invalid new patch list for CVE {cve_id} (not list): {new_patch_list}")
                    continue
                for new_patch in new_patch_list:
                    if not isinstance(new_patch, dict) or "id" not in new_patch or "epoch" not in new_patch:
                        logger.warning(f"Skip invalid new patch for CVE {cve_id} (missing id/epoch): {new_patch}")
                        continue
                    patch_id = new_patch["id"]
                    epoch = new_patch["epoch"]
                    existing_dict[cve_id][patch_id][epoch] = new_patch

        merged = []
        for cve_id, patch_dict in existing_dict.items():
            all_patch_entries: List[Dict[str, Any]] = []
            for patch_id, epoch_dict in patch_dict.items():
                sorted_patches = sorted(epoch_dict.values(), key=lambda x: x["epoch"])
                all_patch_entries.extend(sorted_patches)
            merged.append({cve_id: all_patch_entries})
        return merged

    elif flag == "log":
        merged = {log["vul_id"]: log for log in existing if "vul_id" in log}
        for log in new:
            vul_id = log.get("vul_id")
            if vul_id and vul_id not in merged:
                merged[vul_id] = log
        return list(merged.values())
    else:
        return existing + new

def load_template(template_path: str) -> str:
    """Load prompt template from file."""
    if template_path and Path(template_path).exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Template file not found")

if __name__ == "__main__":
    logger = get_logger()
    try:
        pass
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
