"""E5：真實標籤分類 loss，加上固定 teacher 的蒸餾 loss。"""

import math

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits,
    teacher_logits,
    targets,
    temperature=2.0,
    kd_weight=0.5,
):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必須是有限正數")

    if not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("kd_weight 必須是有限非負數")

    if student_logits.ndim != 2 or student_logits.shape[0] == 0:
        raise ValueError("student_logits 必須是非空的 [B, C]")

    if targets.shape != (student_logits.shape[0],):
        raise ValueError("targets 必須是 [B]")

    if not torch.isfinite(student_logits).all():
        raise ValueError("Student logits 含有 NaN 或 Inf")

    # 明確使用 float32 計算 loss。
    student = student_logits.float()
    ce = F.cross_entropy(student, targets)

    # 對照組完全不需要 teacher。
    if kd_weight == 0:
        return {
            "total": ce,
            "ce": ce,
            "kd": ce.new_zeros(()),
        }

    if teacher_logits is None:
        raise ValueError("啟用蒸餾時，必須提供 teacher_logits")

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher 與 student 的 logits 形狀不一致")

    if teacher_logits.device != student_logits.device:
        raise ValueError("Teacher 與 student logits 必須在相同裝置")

    if not torch.isfinite(teacher_logits).all():
        raise ValueError("Teacher logits 含有 NaN 或 Inf")

    # 即使呼叫端忘記 no_grad，這裡也阻止梯度傳給 teacher。
    teacher = teacher_logits.detach().float()

    student_log_prob = F.log_softmax(
        student / temperature, dim=-1
    )
    teacher_log_prob = F.log_softmax(
        teacher / temperature, dim=-1
    )

    # KL(teacher || student)
    # kd 已包含 T²，但尚未乘上 kd_weight。
    kd = F.kl_div(
        student_log_prob,
        teacher_log_prob,
        reduction="batchmean",
        log_target=True,
    ) * temperature**2

    total = ce + kd_weight * kd

    return {
        "total": total,
        "ce": ce,
        "kd": kd,
    }


@torch.no_grad()
def frozen_teacher_logits(teacher, waveforms):
    """固定 teacher 推論，並保留外部的 PyTorch 亂數狀態。"""
    if any(module.training for module in teacher.modules()):
        raise RuntimeError("Teacher 所有模組都必須處於 eval 模式")

    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("Teacher 所有參數都必須固定")

    device = next(teacher.parameters()).device
    cuda_devices = [device.index] if device.type == "cuda" else []

    # 額外的 teacher 推論不應改變後續裁切／訓練的亂數序列。
    with torch.random.fork_rng(devices=cuda_devices):
        return teacher(waveforms).detach()