"""上傳後核對所有音訊的官方 SHA-256；不修改檔案。"""

import argparse
import hashlib
import json
from pathlib import Path

from data_pipeline.audio_common import read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        if output.exists() or any((args.data_root / name).resolve() in output.parents
                                  for name in ("dataset_A", "dataset_B")):
            parser.error("請使用官方資料夾之外、尚未存在的輸出檔名")
    results = {}
    for name in ("dataset_A", "dataset_B"):
        root = args.data_root / name
        records = [row for split in ("train", "validation", "test") for row in read_records(root, split)]
        failures = []
        for index, row in enumerate(records, start=1):
            try:
                digest = hashlib.sha256()
                with (root / row["audio_path"]).open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != row["sha256"]:
                    failures.append({"sample_id": row["sample_id"], "error": "SHA-256 mismatch"})
            except OSError as exc:
                failures.append({"sample_id": row["sample_id"], "error": str(exc)})
            if index % 200 == 0 or index == len(records):
                print(f"{name}: {index}/{len(records)}, failures={len(failures)}", flush=True)
        results[name] = {"files": len(records), "failures": failures}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if any(result["failures"] for result in results.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
