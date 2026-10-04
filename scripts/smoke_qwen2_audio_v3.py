"""One validation recording; local Qwen2-Audio inference only, no training."""
import argparse
import ast
import copy
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
import time

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
REVISION = "0a095220c30b7b31434169c3086508ef3ea5bf0a"
CLASSES = {
    "dataset_A": ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"],
    "dataset_B": ["Brazil", "Germany", "Italy", "Spain", "UK", "US"],
}


def select_validation(project, dataset):
    directory = (project / dataset).resolve()
    with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "validation"]
    if not rows:
        raise ValueError("No validation rows found; refusing to substitute train or test.")
    row = min(rows, key=lambda r: r["sample_id"])
    relative = Path(row["audio_path"])
    audio = (directory / relative).resolve()
    if relative.is_absolute() or not audio.is_relative_to(directory) or not audio.is_file():
        raise ValueError(f"Invalid audio path: {audio}")
    digest = hashlib.sha256(audio.read_bytes()).hexdigest()
    expected = row.get("sha256", "").strip().lower()
    if not expected or digest != expected:
        raise ValueError("Audio SHA-256 does not match the official manifest.")
    return row["sample_id"], audio, digest


def parse_output(response, classes):
    result = {
        "parser_version": 3, "top3": None, "valid_output": False,
        "parse_error": None, "output_syntax": "unrecognized",
        "raw_json_compliant": False, "format_normalized": False,
    }
    if not isinstance(response, str) or len(response) > 4096:
        result["parse_error"] = "Response must be text of at most 4096 characters."
        return result
    text = response.strip()
    lines = text.splitlines()
    fenced = len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json", "```python"} and lines[-1].strip() == "```"
    if fenced:
        text = "\n".join(lines[1:-1]).strip()
    try:
        labels = json.loads(text)
        result["output_syntax"] = "json_code_fence" if fenced else "json"
        result["raw_json_compliant"] = not fenced
    except json.JSONDecodeError:
        # Accept only a complete three-item array whose delimiters are literal
        # backslash-double-quotes. No global unescaping or scanning prose for labels.
        quoted = r'\\"([^"\\\r\n]*)\\"'
        escaped = re.fullmatch(
            r'\[\s*' + quoted + r'\s*,\s*' + quoted + r'\s*,\s*' + quoted + r'\s*\]',
            text,
        )
        if escaped:
            labels = list(escaped.groups())
            result["output_syntax"] = "escaped_quotes_code_fence" if fenced else "escaped_quotes"
        else:
            try:
                tree = ast.parse(text, mode="eval")
                if not isinstance(tree.body, ast.List) or not all(
                    isinstance(item, ast.Constant) and isinstance(item.value, str)
                    for item in tree.body.elts
                ):
                    raise ValueError("Only a literal list of strings is accepted.")
                labels = ast.literal_eval(tree)
                result["output_syntax"] = "python_list_code_fence" if fenced else "python_list"
            except (SyntaxError, ValueError, TypeError, RecursionError):
                result["parse_error"] = "Expected a JSON array, a literal Python string list, or a three-item escaped-quote array; no surrounding prose."
                return result
    if not isinstance(labels, list) or len(labels) != 3:
        result["parse_error"] = "Expected exactly three labels in a list."
    elif not all(isinstance(x, str) and x in classes for x in labels):
        result["parse_error"] = "An output label is not an exact member of the allowed classes."
    elif len(set(labels)) != 3:
        result["parse_error"] = "Repeated labels are not allowed."
    else:
        result["top3"] = labels
        result["valid_output"] = True
        result["format_normalized"] = result["output_syntax"] != "json"
    return result


def parse_top3(response, classes):
    result = parse_output(response, classes)
    return result["top3"], result["parse_error"]


def reparse_report(source, dataset):
    raw_bytes = source.read_bytes()
    report = json.loads(raw_bytes)
    if report["dataset"] != dataset or report["split"] != "validation":
        raise ValueError("Report dataset/split does not match the requested validation smoke test.")
    if report["model_id"] != MODEL_ID or report["revision"] != REVISION:
        raise ValueError("Unexpected model ID/revision in report.")
    parsed = parse_output(report["raw_response"], CLASSES[dataset])
    report["previous_parse"] = {
        key: report.get(key) for key in ["top3", "valid_output", "parse_error", "parser_version"]
    }
    report.update(parsed)
    report["reparsed_from"] = str(source.resolve())
    report["source_report_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    report["inference_rerun"] = False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = source.with_name(f"parsed_{source.stem}_{stamp}.json")
    with destination.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("Source report:", source)
    print("Raw response:", report["raw_response"])
    print("Top-3:", parsed["top3"])
    print("Output syntax:", parsed["output_syntax"])
    print("Parser version:", parsed["parser_version"])
    print("Raw JSON compliant:", parsed["raw_json_compliant"])
    print("Report:", destination)
    if not parsed["valid_output"]:
        print("FAIL:", parsed["parse_error"])
        return 2
    print("PASS: parsed three distinct allowed labels; no model inference rerun")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("/workspace/HW1_code"))
    parser.add_argument("--dataset", choices=list(CLASSES), default="dataset_A")
    parser.add_argument("--reparse-latest", action="store_true", help="Reparse the latest saved smoke report without loading the model.")
    args = parser.parse_args()
    project = args.project.resolve()
    setup = project / "reports" / "ALM_setup"
    if args.reparse_latest:
        sources = sorted(setup.glob(f"smoke_{args.dataset}_[0-9]*Z.json"))
        if not sources:
            raise FileNotFoundError("No original smoke report found for this dataset.")
        return reparse_report(sources[-1], args.dataset)
    sample_id, audio_path, audio_sha = select_validation(project, args.dataset)
    manifest = json.loads((setup / "model_download.json").read_text(encoding="utf-8"))
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("Unexpected model ID or revision in download manifest.")
    snapshot = Path(manifest["snapshot_path"])
    for name, size in manifest["files"].items():
        p = snapshot / name
        if not p.is_file() or p.stat().st_size != size:
            raise ValueError(f"Missing or wrong-size model file: {p}")

    # Set offline mode before importing the HF libraries. No extra model downloads.
    os.environ["HF_HOME"] = "/workspace/.hf_alm"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    import bitsandbytes as bnb
    from importlib.metadata import version
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment.")
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    waveform, source_sr = sf.read(audio_path, dtype="float32", always_2d=True)
    channels = waveform.shape[1]
    waveform = waveform.mean(axis=1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Empty or non-finite waveform.")
    original_seconds = len(waveform) / source_sr
    target_sr = int(processor.feature_extractor.sampling_rate)
    if source_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=target_sr, res_type="soxr_hq")
    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    max_samples = int(processor.feature_extractor.n_samples)
    if len(waveform) > max_samples:
        raise ValueError("Audio exceeds the processor limit; refusing silent truncation.")

    classes = CLASSES[args.dataset]
    task = "US release decade" if args.dataset == "dataset_A" else "release market of this 1980s recording"
    prompt = (
        f"Listen to the music and predict the {task}. "
        f"Allowed labels: {', '.join(classes)}. "
        "Return exactly three distinct allowed labels, ranked from most likely to least likely. "
        "Return only a JSON array of three strings, with no explanation or other text."
    )
    # The neutral placeholder is rendered to audio tokens; no label or filename is sent.
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "input.wav"},
        {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=rendered, audio=[waveform], sampling_rate=target_sr, return_tensors="pt", padding=True)
    if "input_features" not in inputs or "feature_attention_mask" not in inputs:
        raise RuntimeError("The processor did not produce audio inputs.")
    print(f"Dataset: {args.dataset} | split: validation | sample: {sample_id}", flush=True)
    print(f"Audio: {original_seconds:.2f}s | {source_sr} Hz -> {target_sr} Hz | full recording", flush=True)
    print("Loading local model: NF4 language layers; FP16 audio tower/projector...", flush=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["audio_tower", "multi_modal_projector", "lm_head"],
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(snapshot), local_files_only=True, quantization_config=quantization,
        device_map={"": 0}, dtype=torch.float16, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    model.requires_grad_(False)
    count_4bit = sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    if not count_4bit:
        raise RuntimeError("No 4-bit linear layers were loaded.")
    if any(isinstance(m, bnb.nn.Linear4bit) for m in model.audio_tower.modules()):
        raise RuntimeError("Audio tower was unexpectedly quantized.")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    print(f"Model loaded: {count_4bit} 4-bit layers | {load_seconds:.1f}s", flush=True)
    inputs = {k: v.to(device="cuda", dtype=torch.float16) if v.is_floating_point() else v.to("cuda")
              for k, v in inputs.items()}
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = 1
    generation.temperature = None
    generation.top_p = None
    generation.top_k = None
    generation.max_new_tokens = 96
    generation.use_cache = True
    if processor.tokenizer.pad_token_id is not None:
        generation.pad_token_id = processor.tokenizer.pad_token_id
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=generation)
    torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    response = processor.batch_decode(
        output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed = parse_output(response, classes)
    top3, error = parsed["top3"], parsed["parse_error"]
    report = {
        "purpose": "single-validation-sample smoke test; not an accuracy evaluation",
        "model_id": MODEL_ID, "revision": REVISION, "dataset": args.dataset,
        "split": "validation", "sample_id": sample_id, "audio_sha256": audio_sha,
        "audio_path": str(audio_path), "duration_seconds": original_seconds,
        "source_sample_rate": source_sr, "processor_sample_rate": target_sr,
        "source_channels": channels, "audio_protocol": "entire recording; mean-to-mono; soxr_hq resampling; no crops",
        "prompt": prompt, "rendered_prompt": rendered,
        "quantization": quantization.to_dict(), "quantized_linear_count": count_4bit,
        "attention_implementation": "sdpa", "seed": 42,
        "generation_config": generation.to_dict(), "raw_response": response,
        **parsed,
        "load_seconds": load_seconds, "inference_seconds": inference_seconds,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "versions": {name: version(name) for name in ["torch", "transformers", "bitsandbytes", "accelerate", "librosa", "soundfile"]},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = setup / f"smoke_{args.dataset}_{stamp}.json"
    with destination.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("\nRaw response:", response)
    print("Top-3:", top3)
    print("Output syntax:", parsed["output_syntax"])
    print("Parser version:", parsed["parser_version"])
    print("Raw JSON compliant:", parsed["raw_json_compliant"])
    print(f"Inference: {inference_seconds:.1f}s")
    print(f"Peak GPU allocated: {report['peak_allocated_gib']:.2f} GiB")
    print("Report:", destination)
    print("PASS: full model inference completed")
    if error:
        print("FAIL: output format:", error)
        return 2
    print("PASS: three distinct valid labels parsed (this does not establish accuracy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
