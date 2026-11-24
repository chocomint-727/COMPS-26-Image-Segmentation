import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class EncoderLayer(nn.Module):
    def __init__(self, c_in, c_out, p_dropout=0.05, scale=2):
        super().__init__()
        self.c_stack = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.pool = nn.MaxPool2d(scale, scale)

    def forward(self, x):
        h = checkpoint(self.c_stack, x, use_reentrant=False)
        h = self.dropout(h)
        return self.pool(h), h


class DecoderLayer(nn.Module):
    def __init__(self, c_in, c_skip, c_out, p_dropout=0.1, scale=2):
        super().__init__()

        self.c_stack = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_in, kernel_size=5, padding=2),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.ConvTranspose2d(
            c_in, c_in, kernel_size=2, stride=2
        )

        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x_skip, x_low):
        upsampled = self.upsample(x_low)
        x_combined = torch.cat([x_skip, upsampled], dim=1)

        h = checkpoint(self.c_stack, x_combined, use_reentrant=False)
        return self.dropout(h)


class UNet(nn.Module):
    def __init__(
        self, output_classes, layer_count,
        raw_channels, channel_outs, scale=2
    ):
        super().__init__()
        self.layer_count = layer_count
        self.scale = scale

        if channel_outs is None:
            channel_outs = [64, 96, 128, 128, 96, 64]

        self.first = nn.Sequential(
            nn.Conv2d(
                in_channels=raw_channels,
                out_channels=channel_outs[0],
                kernel_size=7, padding=3,
                stride=1, bias=False
            ),
            nn.BatchNorm2d(channel_outs[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel_outs[0], channel_outs[0],
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channel_outs[0]),
            nn.ReLU(inplace=True)
        )

        self.down = nn.MaxPool2d(self.scale, self.scale)

        self.encoders = nn.ModuleList([
            EncoderLayer(
                channel_outs[i],
                channel_outs[i + 1],
                scale=self.scale
            )
            for i in range(layer_count - 1)
        ])

        self.bottom_layer = nn.Sequential(
            nn.Conv2d(
                channel_outs[layer_count - 1],
                channel_outs[layer_count - 1],
                kernel_size=3, padding=1,
                stride=1, bias=False
            ),
            nn.BatchNorm2d(channel_outs[layer_count - 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel_outs[layer_count - 1],
                channel_outs[layer_count],
                kernel_size=3, padding=1,
                stride=1, bias=False
            ),
            nn.BatchNorm2d(channel_outs[layer_count]),
            nn.ReLU(inplace=True)
        )

        self.decoders = nn.ModuleList([
            DecoderLayer(
                channel_outs[i],
                channel_outs[-i - 1],
                channel_outs[i + 1],
                scale=self.scale
            )
            for i in range(layer_count, len(channel_outs) - 1)
        ])

        self.upscaler = nn.Sequential(
            nn.Conv2d(
                channel_outs[-1],
                channel_outs[-1] * (self.scale ** 2),
                kernel_size=1, stride=1, padding=0
            ),
            nn.PixelShuffle(self.scale)
        )

        self.out = nn.Sequential(
            nn.Conv2d(
                channel_outs[-1] + channel_outs[0],
                channel_outs[-1],
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(channel_outs[-1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                channel_outs[-1],
                output_classes,
                kernel_size=3, padding=1, stride=1
            )
        )

    def forward(self, x):
        first_act = self.first(x)
        activations = [first_act]
        h = self.down(first_act)

        for encoder in self.encoders:
            h, a_map = encoder(h)
            activations.append(a_map)

        up = self.bottom_layer(h)

        for i, decoder in enumerate(self.decoders):
            up = decoder(activations[-i - 1], up)

        up = self.upscaler(up)

        return self.out(
            torch.cat([activations[0], up], dim=1)
        )