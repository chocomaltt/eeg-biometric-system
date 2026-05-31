from torch.nn import Module, LSTM
from torch.nn.modules import Sequential, LazyConv2d, LazyBatchNorm2d, MaxPool2d
from torch.nn.modules.linear import LazyLinear
from torch.nn import SELU, ELU
import torch

class embedding_model(Module):
    def __init__(self):
        super(embedding_model, self).__init__()

        self.input = Sequential(
            LazyConv2d(out_channels=64, kernel_size=1, padding="same"),
            SELU()
        )

        self.conv2_temporal = Sequential(
            LazyConv2d(out_channels=32, kernel_size=4, padding="same"),
            SELU()
        )

        self.batch_normalization = LazyBatchNorm2d(32)

        self.elu = ELU()

        self.MaxPool2d = MaxPool2d(kernel_size=(2, 2))

        self.conv2_spatial = Sequential(
            LazyConv2d(out_channels=64, kernel_size=2, padding="same"),
            SELU()
        )
        
        self.lstm = LSTM(
            input_size=2048,
            hidden_size=128,
            batch_first=True
        )

        self.dense = Sequential(
            LazyLinear(128),
            SELU()
        )

    def forward(self, x):
        if x.dim() == 3:
            # Dataloader output (B, H, W) -> Conv2d expects (B, C, H, W)
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 3D or 4D EEG input, got shape {tuple(x.shape)}")

        if x.size(1) != 1:
            raise ValueError(
                f"Expected input channel dimension C=1 for Conv2d, got shape {tuple(x.shape)}"
            )

        x = self.input(x)
        x = self.conv2_temporal(x)
        x = self.batch_normalization(x) 
        x = self.elu(x)
        x = self.MaxPool2d(x)
        x = self.conv2_spatial(x)
        x = self.elu(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dense(x)
        # norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        # x = x/norm
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x