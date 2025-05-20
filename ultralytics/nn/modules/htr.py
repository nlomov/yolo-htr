import torch
from torch.nn import Module, ModuleList
from torch.nn import Conv2d, InstanceNorm2d, Dropout, Dropout2d, LSTM, ReLU
from torch.nn.functional import pad
from ultralytics.utils.tal import make_anchors

import random


class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class FCN_Encoder(Module):
    def __init__(self, params={'feature_size': 512, "input_channels": 3, "dropout": 0.5, 'v_stride': 1}):
        super(FCN_Encoder, self).__init__()

        self.dropout = params["dropout"]
        self.fs = params.get('feature_size', 256)
        self.half_fs = self.fs // 2
        self.k = params.get('k', 3)
        self.h_stride = params.get('h_stride', 2)
        self.v_stride = params.get('v_stride', 2)
        self.atrous = params.get('atrous', None)

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout, k=self.k, atrous=self.atrous),
            ConvBlock(16, 32, stride=(self.v_stride, self.h_stride), dropout=self.dropout, k=self.k, atrous=self.atrous),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout, k=self.k, atrous=self.atrous),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout, k=self.k, atrous=self.atrous),
            ConvBlock(128, self.half_fs, stride=(2, 1), dropout=self.dropout, k=self.k, atrous=self.atrous),
            ConvBlock(self.half_fs, self.half_fs, stride=(2, 1), dropout=self.dropout, k=self.k, atrous=self.atrous),
        ])
        self.blocks = ModuleList([
            DSCBlock(self.half_fs, self.half_fs, pool=(1, 1), dropout=self.dropout),
            DSCBlock(self.half_fs, self.half_fs, pool=(1, 1), dropout=self.dropout),
            DSCBlock(self.half_fs, self.half_fs, pool=(1, 1), dropout=self.dropout),
            DSCBlock(self.half_fs, self.fs, pool=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        return x


class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                 kernel_size=kernel_size, dilation=dilation, stride=stride, 
                                 padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                 dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)


class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.5, atrous=None):
        super(ConvBlock, self).__init__()

        self.atrous = atrous
        self.activation = activation()
        if atrous:
            self.conv1 = Conv2d(in_channels=in_, out_channels=out_ - out_ // (2*len(atrous)) * len(atrous), kernel_size=k, padding=k // 2)
            self.conv2 = Conv2d(in_channels=out_, out_channels=out_ - out_ // (2*len(atrous)) * len(atrous), kernel_size=k, padding=k // 2)
        else:
            self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
            self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

        if atrous:
            self.atrous_conv = nn.ModuleDict({
                str(curr_atrous): nn.ModuleDict({
                    'conv1': Conv2d(in_channels=in_, out_channels=out_ // (2*len(atrous)), kernel_size=k, 
                                    padding=k // 2 + curr_atrous - 1, dilation=curr_atrous),
                    'conv2': Conv2d(in_channels=out_, out_channels=out_ // (2*len(atrous)), kernel_size=k, 
                                    padding=k // 2 + curr_atrous - 1, dilation=curr_atrous)
                })
                for curr_atrous in atrous
            })

    def forward(self, x):
        pos = random.randint(1, 3)
        
        if self.atrous: x = torch.hstack([self.conv1(x)] + [self.atrous_conv[str(curr_atrous)]['conv1'](x) for curr_atrous in self.atrous])
        else: x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        if self.atrous: x = torch.hstack([self.conv2(x)] + [self.atrous_conv[str(curr_atrous)]['conv2'](x) for curr_atrous in self.atrous])
        else: x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        
        return x


class DSCBlock(Module):

    def __init__(self, in_, out_, pool=(1, 1), activation=ReLU, dropout=0.5):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=pool)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x):
        x0 = x
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
            
        if x0.size() == x.size():
            x = x + x0
            
        return x
    
    
class LineDecoderCTC(Module):
    def __init__(self, input_size, vocab_size, hidden_size=512):
        super(LineDecoderCTC, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.lstm = LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.end_conv = Conv2d(in_channels=self.hidden_size, out_channels=self.vocab_size + 1, kernel_size=1)

    def forward(self, x, h=None, groups=0):
        """
        x (B, C, H, W)
        """
        if x.numel() > 0:
            x, h = self.lstm(x, h)
            x = x.permute(2,1,0)
            out = self.end_conv(x).permute(2,1,0)
            if groups == 0:
                out = torch.nn.functional.log_softmax(out, dim=-1)
            else:
                out1 = torch.nn.functional.log_softmax(out[...,:-groups], dim=-1)
                out2 = torch.nn.functional.log_softmax(out[...,-groups:], dim=-1)
                out = out1.unsqueeze(-2) + out2.unsqueeze(-1)
                out = torch.cat([out[...,:-1].reshape(out.shape[:2]+(out.shape[2]*(out.shape[3]-1),)), out1[...,-1:]], axis=-1)
        else:
            out = torch.zeros((0, x.shape[1], self.end_conv.out_channels), dtype=x.dtype, device=x.device)
        return out