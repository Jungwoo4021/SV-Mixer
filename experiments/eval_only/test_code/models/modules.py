import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


# ---------------------
#   Core Module
# ---------------------
class CustomMLPMixerBlock(nn.Module):
    """Single MLP-Mixer block with Token Mixing and Channel Mixing."""
    def __init__(self, seq_len: int, dim: int, token_mixing_dim: int, channel_mixing_dim: int, group_dim: int, dropout: float = 0.1, last_layer=False, return_hidden=False) -> None:
        super().__init__()
        self.return_hidden = return_hidden
        self.token_mixing = TokenMixingBlock(dim, seq_len, token_mixing_dim, dropout, return_hidden)
        self.channel_mixing = ChannelMixingBlock(seq_len, dim, channel_mixing_dim, group_dim, 0 if last_layer else dropout, return_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, dim)
        if self.return_hidden:
            x, h1 = self.token_mixing(x)
            x, h2 = self.channel_mixing(x)
            return x, h1, h2
        else:
            x = self.token_mixing(x)
            x = self.channel_mixing(x)
            return x

class TokenMixingBlock(nn.Module):
    def __init__(self, dim, seq_len, token_mixing_dim, dropout=0.1, return_hidden=False):
        super(TokenMixingBlock, self).__init__()
        self.return_hidden = return_hidden
        self.norm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        
        self.local_mix1 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.local_mix2 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.linear1 = nn.Linear(seq_len, token_mixing_dim)
        self.linear2 = nn.Linear(token_mixing_dim, seq_len)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # batch, seq_len, channel
        identity = x

        # norm
        x = self.norm(x).transpose(1, 2)
        
        # conv branch
        x1 = self.local_mix1(x)
        x1 = self.gelu(x1)
        x1 = self.local_mix2(x1)

        # linear branch
        x2 = self.linear1(x)
        x2 = self.gelu(x2)
        x2 = self.linear2(x2)

        x = x1 + x2
        h = x

        # aggregation
        x = self.dropout(x)
        x = x.transpose(1, 2)

        x = x + identity

        if self.return_hidden:
            return x, h
        else:
            return x

class ChannelMixingBlock(nn.Module):
    def __init__(self, seq_len, dim, mixing_dim, group_dim, dropout=0.1, return_hidden=False):
        super(ChannelMixingBlock, self).__init__()
        self.return_hidden = return_hidden
        self.groups = dim // group_dim

        # downsample_proc
        self.norm1 = nn.LayerNorm(dim)
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Conv1d(dim, mixing_dim, kernel_size=1, groups=self.groups)
        self.fc2 = nn.Conv1d(mixing_dim, dim, kernel_size=1, groups=self.groups)
        self.upsample = nn.Upsample(size=seq_len, mode='linear', align_corners=False)

        # normal_proc
        self.fc3 = nn.Conv1d(dim, mixing_dim, kernel_size=1, groups=self.groups)
        self.fc4 = nn.Conv1d(mixing_dim, dim, kernel_size=1, groups=self.groups)
       
        # channel mixing
        self.fc5 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=self.groups)
        self.fc6 = nn.Conv1d(dim, dim, kernel_size=1)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.transpose(1, 2)
        x = self.norm1(x).transpose(1, 2)

        # downsample_proc
        x1 = self.downsample(x)
        x1 = self.fc1(x1)
        x1 = self.gelu(x1)
        x1 = self.fc2(x1)
        x1 = self.upsample(x1)

        # normal_proc
        x2 = self.fc3(x)
        x2 = self.gelu(x2)
        x2 = self.fc4(x2)
        
        # channel_mixing
        x = x1 + x2
        x = self.fc5(x)
        x = self.gelu(x)
        x = self.fc6(x)
        h = x
        x = self.dropout(x)
        x = identity + x

        x = x.transpose(1, 2)

        if self.return_hidden:
            return x, h
        else:
            return x
        

# ---------------------
#   Sub Module
# ---------------------
class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1

        bns, convs = [], []
        num_pad = math.floor(kernel_size / 2) * dilation
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x_split = torch.split(x, self.width, dim=1)
        for i in range(self.nums):
            sp = x_split[i] if i == 0 else sp + x_split[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            x = sp if i == 0 else torch.cat((x, sp), dim=1)
        x = torch.cat((x, x_split[self.nums]), dim=1)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.se(x)
        x += identity
        return x

class WavLMGroupNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.activation = nn.GELU(approximate='none')

        self.layer_norm = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels, affine=True)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class WavLMNoLayerNormConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        
        self.activation = nn.GELU(approximate='none')

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states

class WavLMFeatureProjection(nn.Module):
    def __init__(self, conv_dim, hidden_size):
        super().__init__()
        self.layer_norm = nn.LayerNorm(conv_dim, eps=1e-5)
        self.projection = nn.Linear(conv_dim, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class GroupedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, num_groups=4, bidirectional=False):
        super(GroupedGRU, self).__init__()
        self.num_groups = num_groups
        self.group_input_dim = input_dim // num_groups
        self.group_hidden_dim = hidden_dim // num_groups
        self.bidirectional = bidirectional

        self.grus = nn.ModuleList([
            nn.GRU(
                input_size=self.group_input_dim,
                hidden_size=self.group_hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional
            ) for _ in range(num_groups)
        ])

    def forward(self, x):
        group_inputs = x.chunk(self.num_groups, dim=2)
        group_outputs = []

        for i, group_input in enumerate(group_inputs):
            out, _ = self.grus[i](group_input)
            group_outputs.append(out)

        out = torch.cat(group_outputs, dim=2)
        return out
        
class MultiscaleProcessingBlock(nn.Module):
    def __init__(self, ratio, seq_len, dim, group_dim, last_gelu=True):
        super(MultiscaleProcessingBlock, self).__init__()
        self.groups = dim // group_dim
        self.last_gelu = last_gelu

        self.downsample = nn.AvgPool1d(kernel_size=ratio, stride=ratio)
        self.upsample = nn.Upsample(size=seq_len, mode='linear', align_corners=False)

        # Grouped 1x1 Convolutions
        self.fc1 = nn.Conv1d(dim, dim, kernel_size=1, groups=self.groups)
        self.fc2 = nn.Conv1d(dim, dim, kernel_size=1, groups=self.groups)

        self.gelu = nn.GELU()

    def forward(self, x, x_down=None):
        # Downsample path
        if x_down is None:
            x_down = self.downsample(x)
        x_down = self.fc1(x_down)

        # Original path
        x_orig = self.fc2(x)

        # Fusion with cross-scale information
        x_fused = x_orig + self.upsample(x_down)
        x_down_fused = x_down + self.downsample(x)

        # Activation
        if self.last_gelu:
            x_fused = self.gelu(x_fused)
            x_down_fused = self.gelu(x_down_fused)

        return x_fused, x_down_fused