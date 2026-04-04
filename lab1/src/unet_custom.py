"""Имплементация архитектуры U-Net с нуля (пункт 4 лабораторной работы)."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Двойной свёрточный блок: (Conv2d -> BN -> ReLU) × 2.

    Args:
        in_channels: Количество входных каналов.
        out_channels: Количество выходных каналов.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Прямой проход через двойную свёртку.

        Args:
            x: Входной тензор [B, C_in, H, W].

        Returns:
            Тензор [B, C_out, H, W].
        """
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Блок энкодера: MaxPool -> DoubleConv.

    Args:
        in_channels: Количество входных каналов.
        out_channels: Количество выходных каналов.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        """Прямой проход: понижение разрешения + двойная свёртка.

        Args:
            x: Входной тензор [B, C_in, H, W].

        Returns:
            Тензор [B, C_out, H/2, W/2].
        """
        return self.pool_conv(x)


class UpBlock(nn.Module):
    """Блок декодера: UpConv -> Конкатенация со skip-connection -> DoubleConv.

    Args:
        in_channels: Количество входных каналов (от предыдущего слоя декодера).
        out_channels: Количество выходных каналов.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        """Прямой проход: апсемплинг + конкатенация + двойная свёртка.

        Args:
            x: Тензор от предыдущего слоя декодера [B, C_in, H, W].
            skip: Тензор skip-connection от энкодера [B, C_in//2, 2H, 2W].

        Returns:
            Тензор [B, C_out, 2H, 2W].
        """
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class CustomUNet(nn.Module):
    """U-Net архитектура для бинарной сегментации, реализованная с нуля.

    Состоит из энкодера (4 уровня понижения разрешения),
    боттлнека и декодера (4 уровня повышения разрешения)
    со skip-connections между соответствующими уровнями.

    Args:
        in_channels: Количество каналов входного изображения (3 для RGB).
        out_channels: Количество классов на выходе (1 для бинарной сегментации).
        features: Список размеров фильтров для каждого уровня энкодера.
    """

    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Энкодер
        self.encoder_input = DoubleConv(in_channels, features[0])
        self.encoder_blocks = nn.ModuleList([
            DownBlock(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])

        # Боттлнек
        self.bottleneck = DownBlock(features[-1], features[-1] * 2)

        # Декодер
        self.decoder_blocks = nn.ModuleList()
        reversed_features = list(reversed(features))
        in_ch = features[-1] * 2
        for f in reversed_features:
            self.decoder_blocks.append(UpBlock(in_ch, f))
            in_ch = f

        # Выходной слой
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """Прямой проход через U-Net.

        Args:
            x: Входной тензор [B, 3, H, W], H и W должны быть кратны 16.

        Returns:
            Тензор [B, out_channels, H, W] — предсказанные маски (logits).
        """
        # Энкодер
        skip_connections = []
        x = self.encoder_input(x)
        skip_connections.append(x)

        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)

        # Боттлнек
        x = self.bottleneck(x)

        # Декодер
        skip_connections = list(reversed(skip_connections))
        for i, block in enumerate(self.decoder_blocks):
            x = block(x, skip_connections[i])

        return self.final_conv(x)


if __name__ == "__main__":
    model = CustomUNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
