import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = (self.bn1(self.conv1(x))).relu()
        x = (self.bn2(self.conv2(x))).relu()
        x = x + shortcut
        return x


channels = [8, 32, 64]


class SpatialEncoder(nn.Module):
    def __init__(self, args, n_resblocks = 4, channels = channels, in_dim=15 * 15, out_dim=256, seq_len=512):
        super().__init__()
        self.self_group_id = args.self_group_id
        self.conv1 = nn.Conv2d(
            channels[0], channels[1], kernel_size=1, stride=1, padding=0, bias=True
        )  # increase dim of input data
        self.conv2 = nn.Conv2d(
            channels[1], channels[2], kernel_size=1, stride=1, padding=0, bias=True
        )  # increase dim of input data
        self.resblock_stack = nn.Sequential(
            *[ResBlock(channels[2], channels[2], downsample=False) for _ in range(n_resblocks)]
        )
        self.seq_len = seq_len
        self.fc = nn.Linear(in_dim * channels[-1], out_dim)
        self.out_dim = out_dim

    def forward(self, list_of_input, group_id):
        # list_of_input: [number of sample I's alive agents, input channel, h, w ]
        batch_size = len(list_of_input)
        x = torch.cat(list_of_input, 0)
        n_alive_agents_per_sample = [len(i) for i in list_of_input]

        # x: number of sample0's alive agents + number of sample1's alive agents + ..., input channel, h, w
        x = self.conv1(x)  # ch -> 32
        x = self.conv2(x)  # ch -> 64
        x = self.resblock_stack(x)  # share neighbor features
        x = x.reshape(x.shape[0], -1)  # flatten
        x = self.fc(x)  # reduce dim -> 32
        packed_output = x.relu()

        padded_output = self.pad_to_seq_len_with_zeros(packed_output, batch_size, group_id, self.seq_len)

        # list_of_output = []
        # cnt = 0
        # for i in n_alive_agents_per_sample:
        #     list_of_output.append(x[cnt:cnt+i])
        #     cnt += i
        # return list_of_output, padded_output

        return padded_output

    def pad_to_seq_len_with_zeros(self, packed_output, batch_size, group_id, seq_len):
        # packed_output: number of sample0's alive agents + number of sample1's alive agents + ..., output_dim
        # group_id: batch_size, seq_len
        padded = torch.zeros(batch_size, seq_len, packed_output.shape[-1])

        idx = (group_id == self.self_group_id).nonzero()
        padded[idx[:, 0], idx[:, 1]] = packed_output
        return padded
