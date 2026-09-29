"""整理 E2 各層結果，輸出比較圖與 CSV。"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def load_summary(path):
    results = json.loads(path.read_text(encoding="utf-8"))

    if (
        len(results) != 12
        or {item["layer"] for item in results} != set(range(1, 13))
    ):
        raise ValueError(f"{path} 必須包含第 1～12 層各一筆結果")

    for item in results:
        if not 0 <= item["top1"] <= 1:
            raise ValueError("Top-1 必須介於 0 和 1")
        if not 0 <= item["top3"] <= 1:
            raise ValueError("Top-3 必須介於 0 和 1")

    return sorted(results, key=lambda item: item["layer"])


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "reports" / "E2_layer_comparison"

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} 已存在，請先更換輸出資料夾名稱"
        )

    summaries = {
        "dataset_A": load_summary(
            root / "outputs/checkpoints/A_E2_seed42/layer_summary.json"
        ),
        "dataset_B": load_summary(
            root / "outputs/checkpoints/B_E2_seed42/layer_summary.json"
        ),
    }

    output_dir.mkdir(parents=True)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        sharey=True,
    )

    csv_rows = []

    for ax, (dataset_name, results) in zip(axes, summaries.items()):
        layers = [item["layer"] for item in results]
        top1 = [item["top1"] * 100 for item in results]
        top3 = [item["top3"] * 100 for item in results]

        # 和訓練程式使用相同的選層規則。
        selected = min(
            results,
            key=lambda item: (
                -item["top1"],
                item["loss"],
                item["layer"],
            ),
        )

        ax.plot(
            layers,
            top1,
            marker="o",
            color="#0072B2",
            label="Top-1",
        )
        ax.plot(
            layers,
            top3,
            marker="s",
            color="#D55E00",
            label="Top-3",
        )

        selected_layer = selected["layer"]
        selected_top1 = selected["top1"] * 100

        ax.scatter(
            [selected_layer],
            [selected_top1],
            marker="*",
            s=180,
            color="#009E73",
            zorder=5,
            label="Selected by Top-1",
        )

        ax.annotate(
            f"Layer {selected_layer}: {selected_top1:.2f}%",
            xy=(selected_layer, selected_top1),
            xytext=(8, 12),
            textcoords="offset points",
        )

        ax.set_title(dataset_name)
        ax.set_xlabel("MERT Transformer layer")
        ax.set_xticks(layers)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")

        for item in results:
            csv_rows.append(
                {
                    "dataset": dataset_name,
                    "layer": item["layer"],
                    "best_epoch": item["epoch"],
                    "validation_loss": item["loss"],
                    "top1_percent": item["top1"] * 100,
                    "top3_percent": item["top3"] * 100,
                }
            )

        print(
            f"{dataset_name}: "
            f"selected layer={selected_layer}, "
            f"epoch={selected['epoch']}, "
            f"Top-1={selected_top1:.2f}%, "
            f"Top-3={selected['top3']:.2%}"
        )

    axes[0].set_ylabel("Validation accuracy (%)")
    fig.suptitle(
        "E2: Layer-wise probing\n"
        "Each point uses that layer's best validation checkpoint"
    )
    fig.tight_layout()

    fig.savefig(output_dir / "layer_comparison.png", dpi=200)
    fig.savefig(output_dir / "layer_comparison.pdf")
    plt.close(fig)

    with (output_dir / "layer_comparison.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(csv_rows[0]),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print("比較圖與 CSV 已輸出：", output_dir)


if __name__ == "__main__":
    main()