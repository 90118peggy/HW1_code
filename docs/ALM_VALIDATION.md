# Fixed ALM validation protocol (2026-10-04)

This experiment evaluates Qwen/Qwen2-Audio-7B-Instruct on the official validation split only: Dataset A 132 recordings and Dataset B 102 recordings, each with the fixed direct and acoustic prompts (468 primary predictions). It performs inference only. It never reads test audio or trains a model.

The exact prompts, model revision, audio protocol, retry rule, source-code hashes and manifest hashes were frozen before full validation scoring in `ALM_VALIDATION_PROTOCOL_20261004.json`. The direct prompt retains the verified smoke prompt; the acoustic prompt adds a fixed instruction to consider instrumentation, rhythm, vocal style and production sound. Neither contains the sample label, filename or sample ID.

The local model uses NF4 language layers and FP16 audio tower/projector, with greedy decoding and at most 96 generated tokens. Full recordings are averaged to mono and resampled from 24 kHz to the processor's 16 kHz using soxr_hq. Audio exceeding the processor limit is rejected rather than silently truncated. Qwen model revision: `0a095220c30b7b31434169c3086508ef3ea5bf0a`.

## Parsing and scoring

Parser v3 accepts exact three-label JSON arrays, literal Python string lists, or arrays whose delimiters contain escaped quotes, optionally within a supported code fence. Labels must be distinct exact class members and preserve order. Surrounding prose, duplicates, unknown labels and other lengths are invalid. Raw JSON compliance and deterministic formatting normalization are reported separately.

If the initial answer is invalid, exactly one fresh inference uses the same audio and original prompt plus a fixed formatting reminder. Both initial and final answers are preserved and scored separately; retry is never triggered by a wrong class prediction. Invalid predictions receive no guessed fallback and count as incorrect in both Top-1 and Top-3 using the full completed-sample denominator. Confusion matrices have six true-class rows and an explicit seventh INVALID prediction column.

## Run and resume

Use the existing separate environment `/workspace/venvs/hw1-alm`; do not replace the MERT environment. `requirements-alm.txt` records the installed versions. Model files must already be present and described by `reports/ALM_setup/model_download.json`; all model loading is offline. Official datasets and pretrained model weights are not included in Git.

```bash
cd /workspace/HW1_code
/workspace/venvs/hw1-alm/bin/python -m unittest discover -s tests -p test_alm_validation_runner.py -v
/workspace/venvs/hw1-alm/bin/python -m scripts.evaluate_alm_validation --dry-run
/workspace/venvs/hw1-alm/bin/python -u -m scripts.evaluate_alm_validation
```

The third command resumes the same run, skipping completed records. A nonblocking run lock prevents concurrent duplicate inference. Do not launch a second GPU job while the existing process is running. Resume refuses changed code, prompts, manifests or parser results. The four-item pilot used `--max-new-items 4` and those records remain part of the same final evaluation. Each completed prediction is saved atomically under `reports/ALM_validation_20261004/records/`; the run produces protocol, runtime metadata, initial/final metrics, prediction CSV and confusion CSVs.

After the process finishes, audit saved results without inference:

```bash
/workspace/venvs/hw1-alm/bin/python -m scripts.evaluate_alm_validation --summarize-only
/workspace/venvs/hw1-alm/bin/python -m scripts.collect_hw1_evidence
```

The evidence collector reads existing E0-E5 validation metrics and checks confusion-matrix totals and Top-1 reconstruction. These are existing evaluations, not rerun checkpoints. The final ALM result must have `complete: true` and exactly 468 predictions before reporting accuracy. A partial pilot is not an accuracy estimate.

## Limits and attribution

This is a single-seed, quantized-model validation experiment with two prompts fixed before the full evaluation; it is not evidence of test generalization or prompt optimization. Pretraining-data overlap cannot be excluded. The pretrained model is [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct); loading uses Hugging Face Transformers and bitsandbytes. Official data, credentials, pretrained caches and trained checkpoints are excluded from this source commit. No online submission is performed.
