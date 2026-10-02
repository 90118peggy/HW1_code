import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_pipeline.audio_common import AudioConfig
from data_pipeline.waveform_dataset import HW1WaveformDataset
from models.distillation_loss import (
    distillation_loss,
    frozen_teacher_logits,
)
from models.mert_partial_finetune import MERTPartialFinetune
from scripts.check_mert_finetune import (
    state_digest,
    check_group_gradients,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_loss_math():
    """先用小型人工 logits 驗證 loss，不涉及真實資料。"""
    teacher = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]],
        requires_grad=True,
    )
    student = torch.tensor(
        [[0.0, 2.0, -1.0], [1.0, 0.0, -1.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)

    control = distillation_loss(
        student, None, targets, kd_weight=0.0
    )
    torch.testing.assert_close(
        control["total"],
        F.cross_entropy(student, targets),
        rtol=0,
        atol=0,
    )

    same = distillation_loss(
        teacher.detach(),
        teacher.detach(),
        targets,
    )
    if abs(same["kd"].item()) > 1e-6:
        raise RuntimeError("相同 logits 的 KD 應接近 0")

    different = distillation_loss(student, teacher, targets)

    if different["kd"].item() <= 0:
        raise RuntimeError("不同分布的 KD 應大於 0")

    # 單獨使用 KD 反向傳播，排除 CE 的影響。
    different["kd"].backward()

    if student.grad is None or not torch.isfinite(student.grad).all():
        raise RuntimeError("KD 沒有產生有效的 student 梯度")

    if student.grad.abs().sum().item() == 0:
        raise RuntimeError("KD 的 student 梯度全為 0")

    if teacher.grad is not None:
        raise RuntimeError("Teacher 不應收到梯度")

    print("Loss 檢查通過：")
    print("  kd_weight=0 時等於普通 CE")
    print("  相同分布的 KD 接近 0")
    print("  不同分布的 KD 能產生 student 梯度")
    print("  Teacher logits 不會收到梯度")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["A", "B"], required=True)
    args = parser.parse_args()

    check_loss_math()

    if not torch.cuda.is_available():
        raise RuntimeError("未偵測到 CUDA")

    torch.manual_seed(42)
    torch.set_num_threads(2)
    device = torch.device("cuda")

    dataset_name = f"dataset_{args.dataset}"
    feature_layer = {"A": 6, "B": 5}[args.dataset]
    path = (
        PROJECT_ROOT
        / "outputs/checkpoints"
        / f"{args.dataset}_E4_seed42/best.pt"
    )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if (
        checkpoint.get("format_version") != 1
        or checkpoint.get("experiment") != "E4_mert_partial_finetune"
        or checkpoint["dataset"] != dataset_name
        or checkpoint["epoch"] != 0
        or checkpoint["model_config"]["feature_layer"] != feature_layer
        or checkpoint["model_config"]["unfreeze_last_n"] != 2
    ):
        raise RuntimeError(
            "預期讀取保留 E2 起點的 E4 epoch 0 checkpoint"
        )

    dataset = HW1WaveformDataset(
        PROJECT_ROOT / dataset_name,
        split="train",
        config=AudioConfig(**checkpoint["audio_config"]),
    )
    if dataset.class_names != checkpoint["class_names"]:
        raise RuntimeError("類別順序不一致")

    batch = next(iter(DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )))
    waveforms = batch["waveforms"]
    targets = batch["target"].to(device)

    # 分別建立模型，再複製權重。
    # 不共享 teacher 和 student 的 Parameter。
    teacher = MERTPartialFinetune(
        **checkpoint["model_config"]
    ).to(device)
    teacher.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    teacher.requires_grad_(False)
    teacher.eval()

    student = MERTPartialFinetune(
        **checkpoint["model_config"]
    ).to(device)
    student.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    student.train()

    for model in (teacher, student):
        if model.processor.to_dict() != checkpoint["processor_config"]:
            raise RuntimeError("Processor 設定與 checkpoint 不一致")

    teacher_parameters = dict(teacher.named_parameters())
    for name, parameter in student.named_parameters():
        if parameter.data_ptr() == teacher_parameters[name].data_ptr():
            raise RuntimeError(f"Teacher／student 共用了參數：{name}")

    teacher_before = state_digest(teacher)
    if state_digest(student) != teacher_before:
        raise RuntimeError("Teacher／student 初始權重不一致")

    # 確認 helper 不會消耗外部的 CPU／CUDA 亂數序列。
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()

    teacher_logits = frozen_teacher_logits(teacher, waveforms)

    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise RuntimeError("Teacher 推論改變了 CPU 亂數狀態")

    if not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise RuntimeError("Teacher 推論改變了 CUDA 亂數狀態")

    with torch.no_grad():
        initial_student_logits = student(waveforms)

    torch.testing.assert_close(
        initial_student_logits,
        teacher_logits,
        rtol=1e-5,
        atol=1e-6,
    )

    initial = distillation_loss(
        initial_student_logits,
        teacher_logits,
        targets,
    )
    if abs(initial["kd"].item()) > 1e-5:
        raise RuntimeError("初始 teacher／student 的 KD 應接近 0")

    print(f"\nDataset: {dataset_name}")
    print("Teacher checkpoint:", path)
    print("Unfrozen student layers:", student.unfrozen_layers)
    print("Waveforms shape:", tuple(waveforms.shape))
    print(f"Initial KD: {initial['kd'].item():.8f}")
    print("Teacher／student 初始 logits 一致")
    print("Teacher 推論的亂數隔離檢查通過")

    backbone_parameters = [
        parameter
        for parameter in student.encoder.parameters()
        if parameter.requires_grad
    ]
    classifier_parameters = list(student.classifier.parameters())
    trainable = backbone_parameters + classifier_parameters

    optimizer = torch.optim.Adam(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": classifier_parameters, "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )

    torch.cuda.reset_peak_memory_stats(device)

    # 使用同一批 train 音訊做兩步診斷。
    # 第一輪起點相同；第二輪檢查 KD 已能約束更新後的學生。
    for step in (1, 2):
        student.train()
        optimizer.zero_grad(set_to_none=True)

        logits = student(waveforms)
        parts = distillation_loss(
            logits,
            teacher_logits,
            targets,
            temperature=2.0,
            kd_weight=0.5,
        )

        if not all(
            torch.isfinite(value).item() for value in parts.values()
        ):
            raise RuntimeError("Loss 含有 NaN 或 Inf")

        if step == 2:
            kd_gradients = torch.autograd.grad(
                parts["kd"],
                trainable,
                retain_graph=True,
            )
            squared_norm = 0.0
            for gradient in kd_gradients:
                if not torch.isfinite(gradient).all():
                    raise RuntimeError("KD 梯度含有 NaN 或 Inf")
                squared_norm += (
                    gradient.detach().double().square().sum().item()
                )

            kd_norm = squared_norm ** 0.5
            if kd_norm == 0:
                raise RuntimeError("第二步 KD 沒有產生 student 梯度")

            print(f"KD-only gradient norm: {kd_norm:.8f}")
            del kd_gradients

        parts["total"].backward()

        for layer in student.unfrozen_layers:
            check_group_gradients(
                f"Student layer {layer:02d}",
                student.encoder.encoder.layers[layer - 1],
            )
        check_group_gradients("Student classifier", student.classifier)

        for parameter in teacher.parameters():
            if parameter.grad is not None:
                raise RuntimeError("Teacher 收到了梯度")

        for name, parameter in student.named_parameters():
            if not parameter.requires_grad and parameter.grad is not None:
                raise RuntimeError(f"固定的 student 參數有梯度：{name}")

        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        print(
            f"Step {step} | CE={parts['ce'].item():.6f}"
            f" | KD={parts['kd'].item():.6f}"
            f" | total={parts['total'].item():.6f}"
        )

    if state_digest(teacher) != teacher_before:
        raise RuntimeError("Teacher 權重或 buffers 發生改變")

    trainable_names = {
        name
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    changes = {}

    for name, value in student.state_dict().items():
        current = value.detach().cpu()
        original = checkpoint["model_state_dict"][name]

        if not torch.isfinite(current).all():
            raise RuntimeError(f"Student 參數出現 NaN 或 Inf：{name}")

        if name not in trainable_names:
            if not torch.equal(current, original):
                raise RuntimeError(f"固定參數或 buffer 被修改：{name}")
        else:
            changes[name] = (current - original).abs().max().item()

    groups = [
        (
            f"Student layer {layer:02d}",
            f"encoder.encoder.layers.{layer - 1}.",
        )
        for layer in student.unfrozen_layers
    ]
    groups.append(("Student classifier", "classifier."))

    for label, prefix in groups:
        change = max(
            value
            for name, value in changes.items()
            if name.startswith(prefix)
        )
        if change == 0:
            raise RuntimeError(f"{label} 沒有更新")
        print(f"{label} max weight change: {change:.8f}")

    print(
        "Peak allocated GPU memory:",
        f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB",
    )
    print("\nTeacher 無梯度，所有權重與 buffers 保持不變")
    print("Student 指定區塊與分類器成功更新")
    print("Student 其他參數與 buffers 保持不變")
    print("KD 項能獨立提供有效梯度")
    print("E5 自我蒸餾基本檢查通過")
    print("本次沒有儲存或覆寫 checkpoint")


if __name__ == "__main__":
    main()