"""Evaluate four existing E5 best checkpoints. No training or weight writes.

Run in the existing remote MERT environment:
python evaluate_e5.py --project-root /workspace/HW1_code \
    --output-dir /workspace/HW1_code/reports/E5_final_eval
"""
import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

RUNS = {
    "A_control": "A_E5_control_seed42",
    "A_kd": "A_E5_kd_seed42",
    "B_control": "B_E5_control_seed42_retry01",
    "B_kd": "B_E5_kd_seed42_retry01",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_metrics(actual, expected):
    for key in ("samples", "top1", "top3"):
        require(actual[key] == expected[key], f"Metric mismatch: {key}")
    require(math.isclose(actual["loss"], expected["loss"], rel_tol=1e-5, abs_tol=1e-6),
            "Validation loss does not reproduce")


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def per_class(matrix, labels):
    rows = []
    for i, label in enumerate(labels):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        rows.append({"class": label, "support": support, "correct": tp,
                     "recall": tp / support if support else None,
                     "precision": tp / predicted if predicted else None})
    return rows


def paired_comparison(control, kd):
    require(control["class_names"] == kd["class_names"], "Paired class order mismatch")
    left = {r["sample_id"]: r for r in control["predictions"]}
    right = {r["sample_id"]: r for r in kd["predictions"]}
    require(set(left) == set(right), "Paired validation IDs differ")
    paired = []
    for sample_id, c in left.items():
        k = right[sample_id]
        require(c["true_label"] == k["true_label"], "Paired labels differ")
        paired.append({"sample_id": sample_id, "true_label": c["true_label"],
                       "control_top1": c["top1_label"], "kd_top1": k["top1_label"],
                       "control_correct_top1": c["correct_top1"], "kd_correct_top1": k["correct_top1"],
                       "control_correct_top3": c["correct_top3"], "kd_correct_top3": k["correct_top3"]})
    counts = {}
    for metric in ("top1", "top3"):
        ckey, kkey = f"control_correct_{metric}", f"kd_correct_{metric}"
        counts[metric] = {
            "both_correct": sum(r[ckey] == 1 and r[kkey] == 1 for r in paired),
            "both_wrong": sum(r[ckey] == 0 and r[kkey] == 0 for r in paired),
            "kd_fixes_control": sum(r[ckey] == 0 and r[kkey] == 1 for r in paired),
            "kd_breaks_control": sum(r[ckey] == 1 and r[kkey] == 0 for r in paired),
        }
        require(sum(counts[metric].values()) == len(paired), "Pair count mismatch")
    by_class = []
    for c, k in zip(control["per_class"], kd["per_class"]):
        by_class.append({"class": c["class"], "support": c["support"],
                         "control_recall": c["recall"], "kd_recall": k["recall"],
                         "delta_correct_kd_minus_control": k["correct"] - c["correct"]})
    return {"samples": len(paired), "correctness_pairs": counts,
            "per_class": by_class, "paired_predictions": paired}


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.project_root.resolve(), args.output_dir.resolve()
    require(not output.exists(), f"Output exists; choose a new folder: {output}")
    for name in ("dataset_A", "dataset_B", "outputs/checkpoints"):
        protected = (root / name).resolve()
        require(output != protected and protected not in output.parents,
                f"Output must be outside {protected}")
    for folder in RUNS.values():
        for filename in ("best.pt", "reload_validation.json", "run_config.json"):
            path = root / "outputs/checkpoints" / folder / filename
            require(path.is_file(), f"Missing file: {path}")

    sys.path.insert(0, str(root))
    import torch
    from torch.utils.data import DataLoader
    from data_pipeline.audio_common import AudioConfig
    from data_pipeline.waveform_dataset import HW1WaveformDataset
    from models.mert_partial_finetune import MERTPartialFinetune
    from scripts.evaluate_mert import evaluate, save_confusion_plot

    require(torch.cuda.is_available(), "CUDA is required; activate the existing MERT environment")
    torch.set_num_threads(2)
    device = torch.device("cuda")
    reports = {}
    for key, folder in RUNS.items():
        letter, variant = key.split("_", 1)
        directory = root / "outputs/checkpoints" / folder
        checkpoint_path = directory / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(checkpoint.get("format_version") == 1 and
                checkpoint.get("experiment") == "E5_mert_self_distillation", "Unsupported checkpoint")
        require(checkpoint["dataset"] == f"dataset_{letter}" and checkpoint["variant"] == variant,
                f"Wrong dataset/variant in {checkpoint_path}")
        run_config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        require(run_config == checkpoint["run_config"], "Checkpoint/run_config mismatch")
        require(run_config["recording_aggregation"] == "mean_logits", "Unexpected aggregation")
        config = AudioConfig(**checkpoint["audio_config"])
        require(config.eval_chunks == 9, "Expected nine validation crops")
        dataset = HW1WaveformDataset(root / checkpoint["dataset"], split="validation", config=config)
        labels = checkpoint["class_names"]
        require(dataset.class_names == labels, "Class order mismatch")
        loader = DataLoader(dataset, batch_size=run_config["val_batch_size"],
                            shuffle=False, num_workers=0, drop_last=False)
        model = MERTPartialFinetune(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        require(model.processor.to_dict() == checkpoint["processor_config"], "Processor mismatch")
        for name, value in model.state_dict().items():
            require(torch.equal(value.detach().cpu(), checkpoint["model_state_dict"][name]),
                    f"Full-state reload mismatch: {name}")
        model.requires_grad_(False)
        model.eval()
        print(f"\n{key} | selected epoch {checkpoint['epoch']} | validation only", flush=True)
        metrics, predictions, confusion = evaluate(model, loader, labels, device,
                                                   run_config["chunk_batch_size"])
        expected_ids = [r["sample_id"] for r in dataset.records]
        actual_ids = [r["sample_id"] for r in predictions]
        require(len(actual_ids) == len(set(actual_ids)) == len(expected_ids) == len(set(expected_ids)),
                "Duplicate or missing validation ID")
        require(set(actual_ids) == set(expected_ids), "Validation ID set mismatch")
        matrix = confusion.tolist()
        require(int(confusion.sum()) == len(dataset), "Confusion total mismatch")
        for i, label in enumerate(labels):
            require(sum(matrix[i]) == sum(r["true_label"] == label for r in predictions),
                    "Confusion class support mismatch")
        require(sum(matrix[i][i] for i in range(len(labels))) ==
                sum(r["correct_top1"] for r in predictions), "Confusion diagonal mismatch")
        for metric in ("top1", "top3"):
            require(sum(r[f"correct_{metric}"] for r in predictions) / len(predictions) == metrics[metric],
                    "Prediction accuracy mismatch")
        check_metrics(metrics, checkpoint["validation"])
        check_metrics(metrics, json.loads((directory / "reload_validation.json").read_text(encoding="utf-8")))
        report = {"dataset": checkpoint["dataset"], "variant": variant, "split": "validation",
                  "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                  "checkpoint_epoch": checkpoint["epoch"], "class_names": labels,
                  "aggregation": "nine-crop mean logits", **metrics,
                  "confusion_matrix": matrix, "per_class": per_class(matrix, labels),
                  "predictions": predictions}
        run_output = output / key
        run_output.mkdir(parents=True, exist_ok=False)
        save_json(run_output / "metrics.json", {k: v for k, v in report.items() if k != "predictions"})
        write_csv(run_output / "predictions.csv", predictions)
        write_csv(run_output / "per_class.csv", report["per_class"])
        save_confusion_plot(confusion, labels,
                            f"E5 {checkpoint['dataset']} | {variant} | epoch {checkpoint['epoch']}",
                            run_output / "confusion_matrix.png")
        reports[key] = report
        print(f"PASS: full reload, sample coverage, saved metrics | Top-1={metrics['top1']:.2%} | Top-3={metrics['top3']:.2%}", flush=True)
        del checkpoint, model, loader, dataset, confusion
        gc.collect()
        torch.cuda.empty_cache()

    pairs = {}
    for letter in ("A", "B"):
        pairs[letter] = paired_comparison(reports[f"{letter}_control"], reports[f"{letter}_kd"])
        write_csv(output / f"{letter}_paired_predictions.csv", pairs[letter]["paired_predictions"])
        write_csv(output / f"{letter}_per_class_comparison.csv", pairs[letter]["per_class"])
    save_json(output / "E5_evaluation_summary.json", {
        "status": "all_four_runs_verified", "runs": reports, "paired_comparisons": pairs,
        "scope": "Validation only; no training, no test-set evaluation, no checkpoint modification.",
    })
    print(f"\nAll four evaluations passed. Upload: {output / 'E5_evaluation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
