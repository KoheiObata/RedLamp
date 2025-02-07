import torch
import torch.nn as nn
import torch.nn.functional as F


#########################################################################################################
# CONVOLUTIONAL AUTOENCODER
#########################################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, dropout=0.2, normalization='none'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.conv, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        self.convtraspose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                               output_padding=output_padding,
                                               padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.convtraspose, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        num_blocks = len(num_channels)
        layers = []
        for i in range(num_blocks):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dropout=dropout,
                          normalization=normalization)]

        self.network = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv1d(num_channels[-1], embedding_dim, 1)

    def forward(self, x):
        x = self.network(x.transpose(2, 1))
        x = F.max_pool1d(x, kernel_size=x.data.shape[2])
        x = self.conv1x1(x)
        return x


def conv_out_len(seq_len, ker_size, stride, padding, dilation, stack):
    for _ in range(stack):
        seq_len = int((seq_len + 2 * padding - dilation * (ker_size - 1) - 1) / stride + 1)
    return seq_len


class ConvDecoder(nn.Module):
    def __init__(self, embedding_dim, num_channels, seq_len, out_dimension, kernel_size, stride=2, padding=0,
                 dropout=0.2, normalization='none'):
        super().__init__()

        num_channels = num_channels[::-1]
        num_blocks = len(num_channels)

        self.compressed_len = conv_out_len(seq_len, kernel_size, stride, padding, 1, num_blocks)

        # Pad sequence to match encoder length
        if stride > 1:
            output_padding = []
            seq = seq_len
            for _ in range(num_blocks):
                output_padding.append(seq % 2)
                seq = conv_out_len(seq, kernel_size, stride, padding, 1, 1)
            # bit flip
            if kernel_size % 2 == 1:
                output_padding = [1 - x for x in output_padding[::-1]]
            else:
                output_padding = output_padding[::-1]
        else:
            output_padding = [0] * num_blocks

        layers = []
        for i in range(num_blocks):
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvTransposeBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding[i], dropout=dropout, normalization=normalization)]
        self.network = nn.Sequential(*layers)
        self.upsample = nn.Linear(1, self.compressed_len)
        self.conv1x1 = nn.Conv1d(num_channels[-1], out_dimension, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.network(x)
        x = self.conv1x1(x)
        return x.transpose(2, 1)

