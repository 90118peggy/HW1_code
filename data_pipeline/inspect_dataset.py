"""檢查 HW1 官方 manifest；不修改切分、標籤或音訊。僅使用 Python 標準函式庫。"""

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


SPLITS = ("train", "validation", "test")
LABELS = {
    "dataset_A": {"1960s", "1970s", "1980s", "1990s", "2000s", "2010s"},
    "dataset_B": {"Brazil", "Germany", "Italy", "Spain", "UK", "US"},
}
REQUIRED_COLUMNS = {
    "sample_id", "split", "label", "audio_path",
    "duration_seconds", "sample_rate", "sha256",
}


def inspect_dataset(dataset_dir):
    """每列代表一筆錄音；split 是官方分組，不是我們重新分配的結果。"""
    manifest = dataset_dir / "manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(columns)
        if missing or len(columns) != len(set(columns)):
            raise ValueError(f"{manifest}: 欄位缺少 {sorted(missing)}，或有重複欄名")
        rows = list(reader)

    errors = []
    malformed = [i for i, row in enumerate(rows, start=2)
                 if None in row or any(value is None for value in row.values())]
    if malformed:
        raise ValueError(f"{manifest}: CSV 欄數不符，行號 {malformed[:5]}")
    if not rows:
        errors.append("manifest 沒有任何資料列")

    # Counter 計次；set 去除重複後，才能用交集檢查分組是否重疊。
    id_counts = Counter(row["sample_id"] for row in rows)
    duplicate_ids = {key: count for key, count in id_counts.items() if count > 1}
    if duplicate_ids:
        errors.append(f"有 {len(duplicate_ids)} 個重複 sample_id")

    by_split = {split: [row for row in rows if row["split"] == split]
                for split in SPLITS}
    unexpected_splits = sorted({row["split"] for row in rows} - set(SPLITS))
    if unexpected_splits:
        errors.append(f"出現未知 split: {unexpected_splits}")

    split_stats = {}
    for split, records in by_split.items():
        labels = Counter(row["label"] for row in records if row["label"])
        missing_labels = sum(row["label"] == "" for row in records)
        split_stats[split] = {
            "samples": len(records), "labeled_samples": sum(labels.values()),
            "missing_labels": missing_labels, "classes": dict(sorted(labels.items())),
        }
        if not records:
            errors.append(f"{split} 沒有資料")
        if split != "test" and missing_labels:
            errors.append(f"{split} 有 {missing_labels} 筆缺少標籤")
        if split == "test" and labels:
            errors.append("test 出現非空標籤，與官方 README 的隱藏標籤設計不符")
        unknown_labels = set(labels) - LABELS[dataset_dir.name]
        if unknown_labels:
            errors.append(f"{split} 有未知標籤: {sorted(unknown_labels)}")

    split_ids = {split: {row["sample_id"] for row in records}
                 for split, records in by_split.items()}
    overlaps = {f"{a} / {b}": sorted(split_ids[a] & split_ids[b])
                for a, b in combinations(SPLITS, 2)}
    if any(overlaps.values()):
        errors.append("不同 split 的 sample_id 有重疊")

    missing_audio, invalid_rows = [], []
    resolved_paths = []
    for line, row in enumerate(rows, start=2):
        if not row["sample_id"] or any(value != value.strip() for value in row.values()):
            invalid_rows.append(line)

        # audio_path 相對於各 dataset 資料夾，而不是目前的終端機目錄。
        relative = Path(row["audio_path"])
        audio = (dataset_dir / relative).resolve()
        if not row["audio_path"] or relative.is_absolute() or dataset_dir not in audio.parents:
            errors.append(f"第 {line} 行 audio_path 不在資料集內")
            continue
        resolved_paths.append(str(audio))
        if not audio.is_file():
            missing_audio.append(row["audio_path"])
    if invalid_rows:
        errors.append(f"{len(invalid_rows)} 行有空 ID 或欄位前後空白")
    if missing_audio:
        errors.append(f"找不到 {len(missing_audio)} 筆對應音訊")
    duplicate_paths = sum(count - 1 for count in Counter(resolved_paths).values() if count > 1)
    if duplicate_paths:
        errors.append(f"音訊路徑重複引用 {duplicate_paths} 次")

    return {
        "dataset": dataset_dir.name,
        "columns": columns,
        "total_samples": len(rows),
        "splits": split_stats,
        "duplicate_ids": duplicate_ids,
        "split_id_overlaps": overlaps,
        "unexpected_splits": unexpected_splits,
        "missing_audio": missing_audio,
        "invalid_row_numbers": invalid_rows,
        "duplicate_audio_path_references": duplicate_paths,
        "manifest_duration_seconds": dict(Counter(row["duration_seconds"] for row in rows)),
        "manifest_sample_rates": dict(Counter(row["sample_rate"] for row in rows)),
        "artist_overlap_check": "無法驗證：公開 manifest 沒有歌手資訊；保留官方切分。",
        "audio_check_scope": "只檢查檔案存在；尚未解碼音訊、核對 WAV 標頭或計算 SHA-256。",
        "errors": errors,
    }


def print_summary(result):
    print(f"\n{result['dataset']}: 共 {result['total_samples']} 筆")
    for split, stats in result["splits"].items():
        print(f"  {split}: {stats['samples']} 筆，"
              f"有標籤 {stats['labeled_samples']}，空標籤 {stats['missing_labels']}")
        if stats["classes"]:
            print("    " + ", ".join(f"{label}={count}" for label, count in stats["classes"].items()))
    print(f"  重複 ID: {len(result['duplicate_ids'])}")
    for pair, ids in result["split_id_overlaps"].items():
        print(f"  {pair} ID 重疊: {len(ids)}")
    print(f"  缺少音訊: {len(result['missing_audio'])}")
    print(f"  {result['artist_overlap_check']}")
    print(f"  {result['audio_check_scope']}")
    for error in result["errors"]:
        print(f"  ERROR: {error}")
    print("  結果: " + ("未通過" if result["errors"] else "上述檢查通過"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="包含 dataset_A 與 dataset_B 的資料夾，預設為專案根目錄")
    parser.add_argument("--output", type=Path, help="選填：將檢查結果另存為 JSON（不覆寫既有檔案）")
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    results = []
    for name in LABELS:
        try:
            result = inspect_dataset(data_root / name)
        except (OSError, ValueError, csv.Error) as exc:
            print(f"ERROR: {name}: {exc}")
            results.append({"dataset": name, "errors": [str(exc)]})
            continue
        print_summary(result)
        results.append(result)

    if args.output:
        output = args.output.resolve()
        if any(output == data_root / name or data_root / name in output.parents for name in LABELS):
            parser.error("報告必須寫在 dataset_A／dataset_B 之外，保護官方資料。")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output.open("x", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except FileExistsError:
            parser.error("報告已存在；請使用新的 --output 檔名。")
        print(f"\n檢查報告: {output}")
    return 1 if any(result["errors"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
