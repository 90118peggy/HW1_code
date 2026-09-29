"""比對實際載入的 MERT 與官方 checkpoint，不修改模型或權重檔。"""

import torch
from huggingface_hub import hf_hub_download

from models.mert_classifier import (
    FrozenMERTClassifier,
    MODEL_NAME,
    MODEL_REVISION,
)


def main():
    torch.set_num_threads(2)

    # 讀取先前已下載的同一版本 checkpoint。
    # local_files_only=True：這次只找本機快取。
    checkpoint_path = hf_hub_download(
        repo_id=MODEL_NAME,
        filename="pytorch_model.bin",
        revision=MODEL_REVISION,
        local_files_only=True,
    )

    official = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # 使用目前實驗完全相同的模型載入方式。
    # 這次只比對權重，因此留在 CPU 即可。
    model = FrozenMERTClassifier(
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
    )
    actual = model.encoder.state_dict()

    # 只轉換官方已知的兩個舊名稱，不修改數值。
    prefix = "encoder.pos_conv_embed.conv."
    key_mapping = {
        prefix + "weight_g":
            prefix + "parametrizations.weight.original0",
        prefix + "weight_v":
            prefix + "parametrizations.weight.original1",
    }

    expected = {}

    for old_name, tensor in official.items():
        new_name = key_mapping.get(old_name, old_name)

        if new_name in expected:
            raise RuntimeError(f"名稱轉換後重複：{new_name}")

        expected[new_name] = tensor

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))

    print("\n官方有、模型沒有的參數：", missing)
    print("模型有、官方沒有的參數：", extra)

    mismatched = []

    for name in sorted(set(expected) & set(actual)):
        source = expected[name]
        loaded = actual[name]

        if source.shape != loaded.shape:
            mismatched.append(name)
            print(
                f"形狀不一致：{name}\n"
                f"  官方：{tuple(source.shape)}\n"
                f"  模型：{tuple(loaded.shape)}"
            )
            continue

        if source.dtype != loaded.dtype or not torch.equal(source, loaded):
            mismatched.append(name)
            difference = (
                source.to(torch.float64) - loaded.to(torch.float64)
            ).abs().max().item()

            print(
                f"數值或 dtype 不一致：{name}\n"
                f"  官方 dtype：{source.dtype}\n"
                f"  模型 dtype：{loaded.dtype}\n"
                f"  最大絕對差：{difference}"
            )

    print("比對的官方 tensor 數量：", len(expected))
    print("不一致的 tensor 數量：", len(mismatched))

    if missing or extra or mismatched:
        raise RuntimeError(
            "權重比對未通過，請回傳以上結果，先不要開始正式訓練。"
        )

    print("全部 MERT 權重與官方 checkpoint 完全一致")
    print("權重載入檢查通過")


if __name__ == "__main__":
    main()
    