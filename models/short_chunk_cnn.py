import torch
from torch import nn

class ConvBlock(nn.Module):
    """CNN -> BatchNorm -> ReLU -> Max Pool"""

    def __init__(self, in_channels, out_channels): 
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class ShortChunkCNN(nn.Module):
    """輸入標準化頻譜，輸出每段音訊的六類 logits。"""

    def __init__(self, num_classes=6, base_channels=128):
        super().__init__()

        c = base_channels

        self.input_bn = nn.BatchNorm2d(1)

        self.features = nn.Sequential(
            ConvBlock(1, c),
            ConvBlock(c, c),
            ConvBlock(c, c * 2),
            ConvBlock(c * 2, c * 2),
            ConvBlock(c * 2, c * 2),
            ConvBlock(c * 2, c * 2),
            ConvBlock(c * 2, c * 4),
        )

        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(c * 4, c * 4),
            nn.BatchNorm1d(c * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(c * 4, num_classes),
        )

    def forward(self, x):
        x = self.input_bn(x)
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits



if __name__ == "__main__":
    device = torch.device("cuda")

    model = ShortChunkCNN(
        num_classes=6,
        base_channels=128,
    ).to(device)

    model.eval()

    example = torch.randn(2, 1, 128, 346, device=device)

    with torch.no_grad():
        logits = model(example)

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Device:", logits.device)
    print("Input shape:", tuple(example.shape))
    print("Output shape:", tuple(logits.shape))
    print("Parameters:", f"{parameter_count:,}")
    print("Logits:")
    print(logits)

    assert tuple(logits.shape) == (2, 6)
    assert torch.isfinite(logits).all().item()

    print("ShortChunkCNN 形狀與數值檢查通過")