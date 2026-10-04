"""Frozen two-prompt, validation-only Qwen2-Audio evaluation; resumable offline inference."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from datetime import datetime, timezone

from scripts.smoke_qwen2_audio_v3 import CLASSES, MODEL_ID, REVISION, parse_output

COUNTS = {"dataset_A": 132, "dataset_B": 102}
REMINDER = " Format reminder: output ONLY a JSON array containing exactly three distinct allowed label strings in ranked order, without any explanation."


def prompts(dataset):
    task = "US release decade" if dataset == "dataset_A" else "release market of this 1980s recording"
    opening = f"Listen to the music and predict the {task}. "
    ending = (f"Allowed labels: {', '.join(CLASSES[dataset])}. "
              "Return exactly three distinct allowed labels, ranked from most likely to least likely. "
              "Return only a JSON array of three strings, with no explanation or other text.")
    return {
        "direct": opening + ending,
        "acoustic": opening + "Base your prediction on audible characteristics such as instrumentation, rhythm, vocal style, and recording or production sound. " + ending,
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_validation(project):
    data = {}
    for dataset, expected in COUNTS.items():
        directory = (project / dataset).resolve()
        with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = sorted((r for r in csv.DictReader(stream) if r["split"] == "validation"), key=lambda r: r["sample_id"])
        if len(rows) != expected or len({r["sample_id"] for r in rows}) != expected:
            raise ValueError(f"Unexpected validation IDs/count: {dataset}")
        for row in rows:
            if not re.fullmatch(r"[AB]_[A-Za-z0-9]+", row["sample_id"]) or row["label"] not in CLASSES[dataset]:
                raise ValueError("Invalid sample ID or validation label")
            relative = Path(row["audio_path"])
            audio = (directory / relative).resolve()
            if relative.is_absolute() or not audio.is_relative_to(directory) or not audio.is_file():
                raise ValueError(f"Unsafe or missing audio: {audio}")
            if not row.get("sha256") or digest(audio) != row["sha256"].lower():
                raise ValueError(f"Audio checksum mismatch: {audio}")
            row["resolved_audio"] = str(audio)
        data[dataset] = rows
    return data


def metric(records, classes, stage):
    matrix = [[0] * (len(classes) + 1) for _ in classes]
    correct1 = correct3 = valid = raw_json = normalized = 0
    for record in records:
        answer = record[stage]
        truth = record["label"]
        labels = answer["top3"] if answer["valid_output"] else None
        index = classes.index(labels[0]) if labels else len(classes)
        matrix[classes.index(truth)][index] += 1
        valid += bool(labels)
        raw_json += answer["raw_json_compliant"]
        normalized += answer["format_normalized"]
        correct1 += bool(labels) and labels[0] == truth
        correct3 += bool(labels) and truth in labels
    count = len(records)
    return {"samples": count, "top1_correct": correct1, "top3_correct": correct3,
            "top1": correct1 / count if count else None, "top3": correct3 / count if count else None,
            "valid_outputs": valid, "invalid_outputs": count - valid,
            "raw_json_syntax_compliant": raw_json, "format_normalized": normalized,
            "row_labels": classes, "column_labels": classes + ["INVALID"],
            "confusion_matrix_with_invalid": matrix,
            "invalid_scoring": "Invalid predictions count as incorrect in both Top-1 and Top-3; denominator includes all completed recordings."}


def summarize(out, records, protocol_id):
    result = {"protocol_sha256": protocol_id, "split": "validation", "expected_predictions": 468,
              "completed_predictions": len(records), "complete": len(records) == 468, "groups": {}}
    for dataset in COUNTS:
        for prompt_id in prompts(dataset):
            group = [r for r in records if r["dataset"] == dataset and r["prompt_id"] == prompt_id]
            key = f"{dataset}_{prompt_id}"
            result["groups"][key] = {
                "expected": COUNTS[dataset], "complete": len(group) == COUNTS[dataset],
                "initial": metric(group, CLASSES[dataset], "initial"),
                "final": metric(group, CLASSES[dataset], "final"),
                "format_retry_count": sum(r["retry_performed"] for r in group),
                "retry_fixed_count": sum(r["retry_performed"] and r["final"]["valid_output"] for r in group),
            }
    atomic_json(out / "metrics.json", result)
    return result


def load_records(out, protocol_id, data):
    allowed = {(d, p, r["sample_id"]): r for d in data for p in prompts(d) for r in data[d]}
    records = []
    seen = set()
    for path in sorted((out / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (record["dataset"], record["prompt_id"], record["sample_id"])
        if key not in allowed or key in seen or record["protocol_sha256"] != protocol_id:
            raise ValueError(f"Unexpected/duplicate/mismatched existing record: {path}")
        row = allowed[key]
        if record["label"] != row["label"] or record["audio_sha256"] != row["sha256"]:
            raise ValueError(f"Resume data mismatch: {path}")
        for stage in ("initial", "final"):
            parsed = parse_output(record[stage]["raw_response"], CLASSES[key[0]])
            if any(record[stage][k] != v for k, v in parsed.items()):
                raise ValueError(f"Resume parse mismatch: {path}")
        seen.add(key)
        records.append(record)
    return records, seen


def export_csv(out, records, summary):
    fields = ["dataset", "prompt_id", "sample_id", "label", "stage", "top1", "top2", "top3", "valid_output", "raw_json_compliant", "format_normalized", "output_syntax", "parse_error", "raw_response", "retry_performed"]
    target = out / "predictions.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for stage in ("initial", "final"):
                answer = record[stage]
                row = {k: record[k] for k in ("dataset", "prompt_id", "sample_id", "label", "retry_performed")}
                row.update({k: answer[k] for k in fields if k in answer and k != "top3"})
                row.update(dict(zip(("top1", "top2", "top3"), answer["top3"] or ["", "", ""])))
                writer.writerow({**row, "stage": stage})
    os.replace(temporary, target)
    for group, values in summary["groups"].items():
        for stage in ("initial", "final"):
            value = values[stage]
            path = out / f"{group}_{stage}_confusion.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["true_label"] + value["column_labels"])
                writer.writerows([label] + row for label, row in zip(value["row_labels"], value["confusion_matrix_with_invalid"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/workspace/HW1_code"))
    parser.add_argument("--output", type=Path, default=Path("reports/ALM_validation_20261004"))
    parser.add_argument("--max-new-items", type=int, help="Bound this invocation, then resume the SAME frozen run without repeating completed items.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    project = args.project.resolve()
    out = (project / args.output).resolve()
    if not out.is_relative_to(project / "reports"):
        raise ValueError("Output must be inside this project's reports directory")
    if args.max_new_items is not None and args.max_new_items < 1:
        raise ValueError("max-new-items must be positive")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".run.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This exact run is already active; refusing duplicate inference")
    data = read_validation(project)
    model_manifest_path = project / "reports/ALM_setup/model_download.json"
    manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("Unexpected local model ID/revision")
    snapshot = Path(manifest["snapshot_path"])
    for name, size in manifest["files"].items():
        p = snapshot / name
        if not p.is_file() or p.stat().st_size != size:
            raise ValueError(f"Missing/wrong-size model file: {p}")
    protocol = {
        "schema_version": 1, "model_id": MODEL_ID, "revision": REVISION, "split": "validation",
        "expected_counts": COUNTS, "classes": CLASSES, "prompts": {d: prompts(d) for d in COUNTS},
        "prompt_rationale": "direct retains the existing smoke prompt; acoustic adds the same general audible-feature instruction for both datasets, fixed before full validation scoring",
        "format_retry": {"maximum": 1, "trigger": "initial parser valid_output is false", "suffix": REMINDER, "method": "fresh conversation with same audio and original prompt plus format reminder; no label or feedback provided"},
        "invalid_rule": "No fallback or guessed label; invalid counts as wrong with full completed-sample denominator; report initial and final separately.",
        "parser_version": 3, "audio_protocol": "entire recording; mean-to-mono; soxr_hq resampling to processor rate; refuse truncation",
        "seed": 42, "generation_overrides": {"do_sample": False, "num_beams": 1, "temperature": None, "top_p": None, "top_k": None, "max_new_tokens": 96, "use_cache": True},
        "quantization": "NF4 language layers, double quantization, FP16 compute; exclude audio_tower, multi_modal_projector, lm_head",
        "manifest_sha256": {d: digest(project / d / "manifest.csv") for d in COUNTS},
        "model_manifest_sha256": digest(model_manifest_path),
        "runner_sha256": digest(__file__), "smoke_sha256": digest(project / "scripts/smoke_qwen2_audio_v3.py"),
    }
    protocol_id = canonical_hash(protocol)
    config_path = out / "protocol.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Frozen protocol changed; refusing mixed-protocol resume")
    else:
        with config_path.open("x", encoding="utf-8") as stream:
            json.dump(protocol, stream, indent=2, ensure_ascii=False)
    (out / "records").mkdir(exist_ok=True)
    records, seen = load_records(out, protocol_id, data)
    summary = summarize(out, records, protocol_id)
    print(f"PROTOCOL {protocol_id} | existing={len(records)}/468 | output={out}", flush=True)
    if args.dry_run or args.summarize_only or len(records) == 468:
        export_csv(out, records, summary)
        print("PASS: data checksums, model sizes, frozen protocol and resume records verified; no inference", flush=True)
        return 0

    os.environ.update(HF_HOME="/workspace/.hf_alm", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    import bitsandbytes as bnb
    from importlib.metadata import version
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = Qwen2AudioForConditionalGeneration.from_pretrained(str(snapshot), local_files_only=True, quantization_config=quantization,
        device_map={"": 0}, dtype=torch.float16, attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    quantized_count = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    if not quantized_count or any(isinstance(m, bnb.nn.Linear4bit) for m in model.audio_tower.modules()):
        raise RuntimeError("Unexpected quantization topology")
    generation = copy.deepcopy(model.generation_config)
    for key, value in protocol["generation_overrides"].items():
        setattr(generation, key, value)
    if processor.tokenizer.pad_token_id is not None:
        generation.pad_token_id = processor.tokenizer.pad_token_id
    torch.cuda.synchronize()
    runtime = {"protocol_sha256": protocol_id, "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0), "load_seconds": time.perf_counter() - started,
        "quantization": quantization.to_dict(), "quantized_linear_count": quantized_count,
        "generation_config": generation.to_dict(), "versions": {n: version(n) for n in ["torch", "transformers", "bitsandbytes", "accelerate", "librosa", "soundfile"]}}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    atomic_json(out / f"runtime_{stamp}.json", runtime)
    print(f"MODEL READY load_seconds={runtime['load_seconds']:.1f} pid={os.getpid()}", flush=True)

    def infer(waveform, prompt, classes):
        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": "input.wav"}, {"type": "text", "text": prompt}]}]
        rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=rendered, audio=[waveform], sampling_rate=int(processor.feature_extractor.sampling_rate), return_tensors="pt", padding=True)
        if "input_features" not in inputs or "feature_attention_mask" not in inputs:
            raise RuntimeError("Processor missing audio tensors")
        inputs = {k: v.to(device="cuda", dtype=torch.float16) if v.is_floating_point() else v.to("cuda") for k, v in inputs.items()}
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, generation_config=generation)
        torch.cuda.synchronize()
        response = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return {"raw_response": response, **parse_output(response, classes), "inference_seconds": time.perf_counter() - started, "rendered_prompt": rendered}

    # Interleave datasets/prompts in the pilot without changing sample order or prompts.
    tasks = [(d, p, data[d][i]) for i in range(max(COUNTS.values())) for d in COUNTS if i < len(data[d]) for p in prompts(d)]
    measured_start = time.perf_counter()
    new_count = 0
    for dataset, prompt_id, row in tasks:
        key = (dataset, prompt_id, row["sample_id"])
        if key in seen:
            continue
        waveform, source_sr = sf.read(row["resolved_audio"], dtype="float32", always_2d=True)
        channels = waveform.shape[1]
        waveform = waveform.mean(axis=1)
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("Empty/non-finite audio")
        duration = len(waveform) / source_sr
        target_sr = int(processor.feature_extractor.sampling_rate)
        if source_sr != target_sr:
            waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=target_sr, res_type="soxr_hq")
        waveform = np.ascontiguousarray(waveform, dtype=np.float32)
        if len(waveform) > int(processor.feature_extractor.n_samples):
            raise ValueError("Audio exceeds processor limit; refusing truncation")
        prompt = prompts(dataset)[prompt_id]
        initial = infer(waveform, prompt, CLASSES[dataset])
        retry = not initial["valid_output"]
        final = infer(waveform, prompt + REMINDER, CLASSES[dataset]) if retry else initial
        record = {"protocol_sha256": protocol_id, "dataset": dataset, "split": "validation", "prompt_id": prompt_id,
            "sample_id": row["sample_id"], "label": row["label"], "audio_sha256": row["sha256"],
            "duration_seconds": duration, "source_sample_rate": source_sr, "processor_sample_rate": target_sr,
            "source_channels": channels, "initial": initial, "retry_performed": retry, "final": final,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "runtime_file": f"runtime_{stamp}.json"}
        destination = out / "records" / f"{dataset}__{prompt_id}__{row['sample_id']}.json"
        if destination.exists():
            raise RuntimeError("Record unexpectedly exists; refusing overwrite")
        atomic_json(destination, record)
        records.append(record)
        seen.add(key)
        new_count += 1
        seconds_per = (time.perf_counter() - measured_start) / new_count
        print(f"PROGRESS {len(records)}/468 {dataset} {prompt_id} {row['sample_id']} initial_valid={initial['valid_output']} retry={retry} final_valid={final['valid_output']} sec_per_item={seconds_per:.2f} eta_minutes={(468-len(records))*seconds_per/60:.1f}", flush=True)
        if new_count % 10 == 0:
            summarize(out, records, protocol_id)
        if args.max_new_items and new_count >= args.max_new_items:
            break
    summary = summarize(out, records, protocol_id)
    export_csv(out, records, summary)
    atomic_json(out / "progress.json", {"complete": len(records) == 468, "completed": len(records), "expected": 468,
        "protocol_sha256": protocol_id, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(f"FINISHED complete={len(records)==468} records={len(records)}/468", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

