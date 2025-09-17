import torch
import torch.nn as nn

from .modules import *
from data_augmentation import GaussianNoiseAug

class SVMixer(nn.Module):
    def __init__(self, num_hidden_layers, seq_len, hidden_size, dropout=0.1):
        super().__init__()
        
        self.agg_channel = 64
        self.seq_len = seq_len
        self.gaussian = GaussianNoiseAug()

        conv_dim = [512, 512, 512 ,512, 512, 512, 512]
        conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        conv_stride = [5, 2, 2, 2, 2, 2, 2]
        
        self.conv_layers = nn.Sequential(
            *[WavLMGroupNormConvLayer(1, conv_dim[0], conv_kernel[0], conv_stride[0])] 
            + [WavLMNoLayerNormConvLayer(conv_dim[i], conv_dim[i + 1], conv_kernel[i + 1], conv_stride[i + 1]) for i in range(6)]
        )
        self.feat_projection = WavLMFeatureProjection(conv_dim[-1], hidden_size)
        
        # MLP-Mixer encoder
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            CustomMLPMixerBlock(
                seq_len=seq_len,
                dim=hidden_size,
                token_mixing_dim=seq_len * 4,
                channel_mixing_dim=hidden_size * 4,
                group_dim=128,
                dropout=dropout,
                last_layer=(i == num_hidden_layers - 1)
            )
            for i in range(num_hidden_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fbank feature extraction
        x = self.conv_layers(x.unsqueeze(1))
        x = x.transpose(1, 2) 
        x = self.feat_projection(x)

        # MLP-Mixer encoder
        hidden_states_list = []

        # embedding & dropout
        hidden_states = x
        hidden_states = self.dropout(hidden_states)

        # Pass through each MLP-Mixer block and collect hidden states
        for layer in self.layers:
            hidden_states_list.append(hidden_states)
            hidden_states = layer(hidden_states)
        
        hidden_states = self.layer_norm(hidden_states)
        hidden_states_list.append(hidden_states)
        out_hidden_states = torch.stack(hidden_states_list, dim=1)

        return out_hidden_states