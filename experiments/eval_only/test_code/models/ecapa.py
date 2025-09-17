import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformers import WavLMModel

from .modules import Bottle2neck
from ..data_augmentation import GaussianNoiseAug

class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int,
        hidden_size: int,
        channel: int,
        embedding_size: int,
    ) -> None:
        super().__init__()
        
        # Feature aggregation: learnable weighted sum over hidden states
        self.norm = nn.InstanceNorm1d(hidden_size)
        self.w = nn.Parameter(torch.rand(1, num_hidden_layers + 1, 1, 1))
        
        # ECAPA module
        self.conv2 = nn.Conv1d(hidden_size, channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(channel)
        
        self.layer1 = Bottle2neck(channel, channel, kernel_size=3, dilation=2, scale=4)
        self.layer2 = Bottle2neck(channel, channel, kernel_size=3, dilation=3, scale=4)
        self.layer3 = Bottle2neck(channel, channel, kernel_size=3, dilation=4, scale=4)
        self.layer4 = nn.Conv1d(3 * channel, 1536, kernel_size=1)
        
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)
        
        self.gaussian = GaussianNoiseAug()

    def forward(self, x):
        """
        Args:
            x: hidden states from a transformer-based encoder.

        Returns:
            final_embedding: The final embedding tensor.
        """
        # 1. Stack and weighted aggregation
        x = x * self.w.repeat(x.size(0), 1, 1, 1)
        x = x.sum(dim=1)  # (batch, time, features)

        # 2. Instance normalization + augmentation
        x = self.norm(x.transpose(1, 2))  # (batch, features, time)
        if self.training:
            x = self.gaussian(x)

        # 3. ECAPA convolutional front-end
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.layer4(x)
        x = self.relu(x)

        # 4. ASP
        time_steps = x.size(-1)
        mean_stat = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, time_steps)
        std_stat = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, time_steps)
        gx = torch.cat((x, mean_stat, std_stat), dim=1)
        w = self.attention(gx)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(torch.sum((x ** 2) * w, dim=2).sub(mu ** 2).clamp(min=1e-4))

        # 5. Final embedding
        x = torch.cat((mu, sg), dim=1)
        x = self.bn5(x)
        x = self.fc6(x)
        final_embedding = self.bn6(x)

        return final_embedding
