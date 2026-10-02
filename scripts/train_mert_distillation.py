import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
import transformers
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig, training_fingerprint
from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.distillation_loss import (
    distillation_loss,
    frozen_teacher_logits,
)
from models.mert_partial_finetune import MERTPartialFinetune
from scripts.check_mert_finetune import state_digest
from scripts.train_mert import run_epoch
from scripts.train_mert_finetune import (
    seed_everything,
    save_json,
    save_checkpoint,
    check_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_epoch(student, teacher, loader, optimizer, device, settings):
    student.train()

    if student.encoder.training:
        raise RuntimeError("本版 student encoder 應維持 eval 模式")

    trainable = [
        parameter
        for parameter in student.parameters()
        if parameter.requires_grad
    ]

    sums = {"ce": 0.0, "kd": 0.0, "total": 0.0}
    correct1 = 0
    correct3 = 0
    total = 0
    sampling_digest = hashlib.sha256()

    for step, batch in enumerate(loader, start=1):
        waveforms = batch["waveforms"]
        targets = batch["target"].to(device)

        # 記錄順序與裁切起點，供 control／KD 比較。
        sampling_record = {
            "sample_ids": list(batch["sample_id"]),
            "crop_starts": batch["crop_starts"].tolist(),
        }
        sampling_digest.update(
            (
                json.dumps(
                    sampling_record,
                    sort_keys=True,
                    separators=(",", ":"),
                ) + "\n"
            ).encode("utf-8")
        )

        optimizer.zero_grad(set_to_none=True)

        teacher_logits = None
        if teacher is not None:
            teacher_logits = frozen_teacher_logits(teacher, waveforms)

        logits = student(waveforms)
        parts = distillation_loss(
            logits,
            teacher_logits,
            targets,
            temperature=settings["temperature"],
            kd_weight=settings["kd_weight"],
        )

        if not all(
            torch.isfinite(value).item() for value in parts.values()
        ):
            raise RuntimeError("Training loss 含有 NaN 或 Inf")

        parts["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=settings["gradient_clip_norm"],
            error_if_nonfinite=True,
        )
        optimizer.step()

        count = targets.numel()
        total += count

        for key in sums:
            sums[key] += parts[key].item() * count

        top3 = logits.detach().topk(3, dim=1).indices
        correct1 += (top3[:, 0] == targets).sum().item()
        correct3 += (
            top3 == targets[:, None]
        ).any(dim=1).sum().item()

        if step == 1 or step % 20 == 0 or step == len(loader):
            print(
                f"  Train batch {step}/{len(loader)}"
                f" | CE={sums['ce'] / total:.4f}"
                f" | KD={sums['kd'] / total:.4f}"
                f" | total={sums['total'] / total:.4f}",
                flush=True,
            )

    if total == 0:
        raise RuntimeError("Train loader 沒有樣本")

    return {
        "loss": sums["ce"] / total,
        "kd_loss": sums["kd"] / total,
        "total_loss": sums["total"] / total,
        "top1": correct1 / total,
        "top3": correct3 / total,
        "samples": total,
        "sampling_sha256": sampling_digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["A", "B"], required=True)
    parser.add_argument("--mode", choices=["control", "kd"], required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if min(args.epochs, args.patience, args.batch_size) <= 0:
        raise ValueError("epochs、patience、batch-size 必須大於 0")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed 必須介於 0～2**32-1")
    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到 CUDA")

    settings = {
        "temperature": 2.0,
        "kd_weight": 0.0 if args.mode == "control" else 0.5,
        "backbone_lr": 1e-5,
        "classifier_lr": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "val_batch_size": 2,
        "chunk_batch_size": 2,
    }

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(
            f"輸出資料夾已存在：{output}\n請指定新的資料夾。"
        )

    for name in ("dataset_A", "dataset_B"):
        root = (PROJECT_ROOT / name).resolve()
        if output == root or root in output.parents:
            raise ValueError("輸出不能放在官方資料集內")

    seed_everything(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    dataset_name = f"dataset_{args.dataset}"

    source_path = (
        PROJECT_ROOT / "outputs/checkpoints"
        / f"{args.dataset}_E4_seed42/best.pt"
    )
    source = torch.load(
        source_path, map_location="cpu", weights_only=True
    )

    expected_layer = {"A": 6, "B": 5}[args.dataset]
    if (
        source.get("format_version") != 1
        or source.get("experiment") != "E4_mert_partial_finetune"
        or source["epoch"] != 0
        or source["dataset"] != dataset_name
        or source["model_config"]["feature_layer"] != expected_layer
        or source["model_config"]["unfreeze_last_n"] != 2
    ):
        raise RuntimeError("來源必須是已驗證、保留 E2 起點的 E4 epoch 0")

    config = AudioConfig(**source["audio_config"])
    train_data = HW1WaveformDataset(
        PROJECT_ROOT / dataset_name, split="train", config=config
    )
    val_data = HW1WaveformDataset(
        PROJECT_ROOT / dataset_name, split="validation", config=config
    )

    if not (
        train_data.class_names
        == val_data.class_names
        == source["class_names"]
    ):
        raise RuntimeError("類別順序不一致")

    train_generator = torch.Generator()
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=settings["val_batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    student = MERTPartialFinetune(
        **source["model_config"]
    ).to(device)
    student.load_state_dict(source["model_state_dict"], strict=True)

    if student.processor.to_dict() != source["processor_config"]:
        raise RuntimeError("Student processor 設定不一致")

    initial_student_digest = state_digest(student)

    teacher = None
    teacher_digest = None
    if args.mode == "kd":
        teacher = MERTPartialFinetune(
            **source["model_config"]
        ).to(device)
        teacher.load_state_dict(source["model_state_dict"], strict=True)
        teacher.requires_grad_(False)
        teacher.eval()

        if teacher.processor.to_dict() != source["processor_config"]:
            raise RuntimeError("Teacher processor 設定不一致")

        teacher_digest = state_digest(teacher)
        if teacher_digest != initial_student_digest:
            raise RuntimeError("Teacher／student 初始權重不一致")

    print(f"{dataset_name} | mode={args.mode}", flush=True)
    print("訓練前完整重現 E2 起點……", flush=True)

    initial_validation = run_epoch(
        student, val_loader, device, settings["chunk_batch_size"]
    )
    check_metrics(initial_validation, source["validation"])
    print(
        f"起點重現通過 | Top-1={initial_validation['top1']:.2%}"
        f" | Top-3={initial_validation['top3']:.2%}"
    )

    run_config = {
        **vars(args),
        **settings,
        "output_dir": str(output),
        "source_checkpoint": str(source_path),
        "source_sha256": file_sha256(source_path),
        "initial_student_sha256": initial_student_digest,
        "training_fingerprint": training_fingerprint(train_data.records),
        "model_config": source["model_config"],
        "audio_config": source["audio_config"],
        "encoder_mode": "eval_with_grad",
        "recording_aggregation": "mean_logits",
        "sampling_protocol": "epoch_seed=(seed+epoch)%2**32; num_workers=0",
        "torch_version": str(torch.__version__),
        "transformers_version": transformers.__version__,
    }

    metadata = {
        "format_version": 1,
        "experiment": "E5_mert_self_distillation",
        "variant": args.mode,
        "dataset": dataset_name,
        "model_config": source["model_config"],
        "audio_config": source["audio_config"],
        "processor_config": source["processor_config"],
        "class_names": source["class_names"],
        "initial_validation": initial_validation,
        "run_config": run_config,
    }

    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "run_config.json", run_config)
    save_json(output / "initial_validation.json", initial_validation)

    best_path = output / "best.pt"
    last_path = output / "last.pt"
    best_metrics = dict(initial_validation)
    best_epoch = 0
    stale_epochs = 0
    history = []

    save_checkpoint(
        best_path, student, metadata, 0, initial_validation
    )

    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    p for p in student.encoder.parameters()
                    if p.requires_grad
                ],
                "lr": settings["backbone_lr"],
            },
            {
                "params": student.classifier.parameters(),
                "lr": settings["classifier_lr"],
            },
        ],
        weight_decay=settings["weight_decay"],
    )

    for epoch in range(1, args.epochs + 1):
        # 每輪重新固定亂數，避免額外評估影響下一輪抽樣。
        epoch_seed = (args.seed + epoch) % 2**32
        seed_everything(epoch_seed)
        train_generator.manual_seed(epoch_seed)

        start = time.perf_counter()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)

        train_metrics = train_epoch(
            student, teacher, train_loader, optimizer, device, settings
        )
        val_metrics = run_epoch(
            student, val_loader, device, settings["chunk_batch_size"]
        )

        improved = val_metrics["top1"] > best_metrics["top1"]
        tie_better_loss = (
            val_metrics["top1"] == best_metrics["top1"]
            and val_metrics["loss"] < best_metrics["loss"]
        )

        if improved or tie_better_loss:
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            save_checkpoint(
                best_path, student, metadata, epoch, val_metrics
            )
            print("已儲存新的最佳 student")

        stale_epochs = 0 if improved else stale_epochs + 1
        save_checkpoint(
            last_path, student, metadata, epoch, val_metrics
        )

        history.append({
            "epoch": epoch,
            "epoch_seed": epoch_seed,
            "train": train_metrics,
            "validation": val_metrics,
            "best_epoch": best_epoch,
            "seconds": time.perf_counter() - start,
        })
        save_json(output / "history.json", history)

        print(
            f"Train CE={train_metrics['loss']:.4f}"
            f" | KD={train_metrics['kd_loss']:.4f}"
            f" | total={train_metrics['total_loss']:.4f}"
            f" | Top-1={train_metrics['top1']:.2%}"
        )
        print(
            f"Val CE={val_metrics['loss']:.4f}"
            f" | Top-1={val_metrics['top1']:.2%}"
            f" | Top-3={val_metrics['top3']:.2%}"
        )
        print("Sampling SHA256:", train_metrics["sampling_sha256"])

        if stale_epochs >= args.patience:
            print("Validation Top-1 持續未提高，提前停止")
            break

    if teacher is not None:
        if any(p.grad is not None for p in teacher.parameters()):
            raise RuntimeError("Teacher 不應有梯度")
        if state_digest(teacher) != teacher_digest:
            raise RuntimeError("Teacher 權重或 buffers 被修改")
        print("\nTeacher 全程固定檢查通過")

    saved_last = torch.load(
        last_path, map_location="cpu", weights_only=True
    )
    live_state = student.state_dict()
    saved_state = saved_last["model_state_dict"]

    if set(live_state) != set(saved_state):
        raise RuntimeError("最後一輪 checkpoint 的參數鍵不一致")

    for name, value in live_state.items():
        if not torch.equal(value.detach().cpu(), saved_state[name]):
            raise RuntimeError(f"最後一輪權重儲存不一致：{name}")

    print("最後一輪完整 student 儲存檢查通過")

    del live_state, saved_state, saved_last
    del optimizer, student, teacher, source
    gc.collect()
    torch.cuda.empty_cache()

    best = torch.load(
        best_path, map_location="cpu", weights_only=True
    )
    reloaded = MERTPartialFinetune(
        **best["model_config"]
    ).to(device)
    reloaded.load_state_dict(best["model_state_dict"], strict=True)

    if reloaded.processor.to_dict() != best["processor_config"]:
        raise RuntimeError("重新載入的 processor 不一致")

    for name, value in reloaded.state_dict().items():
        if not torch.equal(
            value.detach().cpu(), best["model_state_dict"][name]
        ):
            raise RuntimeError(f"重新載入權重不一致：{name}")

    print("\n重新載入最佳 student，驗證結果……", flush=True)
    reload_metrics = run_epoch(
        reloaded, val_loader, device, settings["chunk_batch_size"]
    )
    check_metrics(reload_metrics, best["validation"])
    save_json(output / "reload_validation.json", reload_metrics)

    print("\nE5 完整 student 重新載入驗證通過")
    print(
        f"mode={args.mode} | 最佳 epoch={best['epoch']}"
        f" | Top-1={reload_metrics['top1']:.2%}"
        f" | Top-3={reload_metrics['top3']:.2%}"
    )
    if best["epoch"] == 0:
        print("最佳模型仍是 E2 起點，本次尚未超越起點。")
    print("最佳模型：", best_path)


if __name__ == "__main__":
    main()