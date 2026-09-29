import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_report(path, expected_dataset):
    report = json.loads(path.read_text(encoding="utf-8"))

    if report["experiment"] != "E3_mert_weighted":
        raise ValueError(f"{path} 不是 E3 評估報告")

    if report["dataset"] != expected_dataset:
        raise ValueError(f"{path} 的資料集名稱不一致")

    if report["split"] != "validation":
        raise ValueError("這張圖應使用 validation 最佳 checkpoint")

    entries = sorted(
        report["layer_weights"],
        key=lambda item: item["layer"],
    )

    if [item["layer"] for item in entries] != list(range(1, 13)):
        raise ValueError("層權重必須完整包含第 1～12 層，且不能重複")

    weights = [float(item["weight"]) for item in entries]

    if not all(math.isfinite(w) and w >= 0 for w in weights):
        raise ValueError("層權重必須是有限、非負的數值")

    if not math.isclose(sum(weights), 1.0, abs_tol=1e-6):
        raise ValueError(f"層權重總和不等於 1：{sum(weights)}")

    return report, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--a-report",
        type=Path,
        default=PROJECT_ROOT / "reports/A_E3_seed42_eval/metrics.json",
    )
    parser.add_argument(
        "--b-report",
        type=Path,
        default=PROJECT_ROOT / "reports/B_E3_seed42_eval/metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports/E3_layer_weights",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()

    for name in ("dataset_A", "dataset_B"):
        dataset_root = (PROJECT_ROOT / name).resolve()
        if output_dir == dataset_root or dataset_root in output_dir.parents:
            raise ValueError("輸出資料夾不能放在官方資料集內")

    if output_dir.exists():
        raise FileExistsError(
            f"輸出資料夾已存在：{output_dir}\n"
            "請用 --output-dir 指定新的資料夾。"
        )

    report_a, weights_a = load_report(args.a_report, "dataset_A")
    report_b, weights_b = load_report(args.b_report, "dataset_B")

    layers = list(range(1, 13))
    uniform_percent = 100 / 12

    # A、B 使用相同的縱軸範圍，方便直接比較。
    largest_percent = max(weights_a + weights_b) * 100
    upper_limit = max(
        20,
        math.ceil((largest_percent + 3) / 5) * 5,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        sharey=True,
    )

    settings = [
        (axes[0], report_a, weights_a, "#2878B5"),
        (axes[1], report_b, weights_b, "#E58B2A"),
    ]

    for ax, report, weights, color in settings:
        percentages = [weight * 100 for weight in weights]

        bars = ax.bar(
            layers,
            percentages,
            color=color,
            width=0.75,
            label="Learned weight",
        )

        ax.axhline(
            uniform_percent,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label=f"Uniform initialization ({uniform_percent:.2f}%)",
        )

        ax.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in percentages],
            padding=3,
            fontsize=8,
        )

        ax.set_title(
            f"{report['dataset']} | "
            f"best epoch {report['checkpoint_epoch']}\n"
            f"Validation Top-1: {report['top1']:.2%}",
            fontsize=12,
        )
        ax.set_xlabel("MERT Transformer layer")
        ax.set_xticks(layers)
        ax.set_ylim(0, upper_limit)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Layer weight (%)")

    fig.suptitle(
        "E3: Learned layer weights at the best validation checkpoint",
        fontsize=14,
    )
    fig.text(
        0.5, 0.02,
        "Weights are mixing coefficients, not standalone layer accuracy "
        "or causal importance.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))

    output_dir.mkdir(parents=True, exist_ok=False)

    fig.savefig(
        output_dir / "layer_weights_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "layer_weights_comparison.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # CSV 同時保留原始比例與百分比。
    with (output_dir / "layer_weights_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "dataset", "checkpoint_epoch", "layer",
            "weight", "weight_percent",
        ])

        for report, weights in [
            (report_a, weights_a),
            (report_b, weights_b),
        ]:
            for layer, weight in enumerate(weights, start=1):
                writer.writerow([
                    report["dataset"],
                    report["checkpoint_epoch"],
                    layer,
                    weight,
                    weight * 100,
                ])

    print(f"圖表與 CSV 已儲存：{output_dir}")


if __name__ == "__main__":
    main()