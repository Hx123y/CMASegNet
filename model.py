import torch
import torch.nn as nn
import torchvision.models as models
from torch import nn, einsum
from einops import rearrange, repeat
# from .Transformer import Transformer
import numpy as np
import torch.nn.functional as F


class model(nn.Module):

    def __init__(self, n_class, training_style='model'):
        super(model, self).__init__()
        self.training_style = training_style
        self.num_resnet_layers = 50
        if self.num_resnet_layers == 18:
            resnet_raw_model1 = models.resnet18(pretrained=True)
            resnet_raw_model2 = models.resnet18(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 34:
            resnet_raw_model1 = models.resnet34(pretrained=True)
            resnet_raw_model2 = models.resnet34(pretrained=True)
            self.inplanes = 512
        elif self.num_resnet_layers == 50:
            resnet_raw_model1 = models.resnet50(pretrained=True)
            resnet_raw_model2 = models.resnet50(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 101:
            resnet_raw_model1 = models.resnet101(pretrained=True)
            resnet_raw_model2 = models.resnet101(pretrained=True)
            self.inplanes = 2048
        elif self.num_resnet_layers == 152:
            resnet_raw_model1 = models.resnet152(pretrained=True)
            resnet_raw_model2 = models.resnet152(pretrained=True)
            self.inplanes = 2048

        ########  Thermal ENCODER  ########

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = BottleStack(dim=1024, fmap_size=(30, 40), dim_out=2048, proj_factor=4,
                                                  num_layers=3, heads=4, dim_head=512)

       
        self.thermal_multiscale_conv1 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.thermal_multiscale_conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)


        self.cross_modal_fusion = nn.Sequential(
            nn.Conv2d(4096, 2048, kernel_size=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
        )


        self.rgb_weight_conv = nn.Conv2d(1024, 1024, kernel_size=1)
        self.thermal_weight_conv = nn.Conv2d(1024, 1024, kernel_size=1)

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = BottleStack(dim=1024, fmap_size=(30, 40), dim_out=2048, proj_factor=4, num_layers=3,
                                              heads=4, dim_head=512)

        self.edge = newEdgeSeg(cin1=256)

        ########  RGB DECODER  ########

        self.deconv5 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2) 
        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2) 
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes // 2, 2,
                                                  stride=2)  
        self.deconv1 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)

        self.skip_tranform = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)


        self.feature_enhance = FeatureEnhanceModule(2048)

    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )

        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, input):

        rgb = input[:, :3]
        thermal = input[:, 3:]

        verbose = False

        # encoder

        ######################################################################
        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if verbose: print("thermal.size() original: ", thermal.size())  # (480, 640)
        ######################################################################

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

        thermal = self.encoder_thermal_conv1(thermal)
        if verbose: print("thermal.size() after conv1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        if verbose: print("thermal.size() after bn1: ", thermal.size())  # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        if verbose: print("thermal.size() after relu: ", thermal.size())  # (240, 320)

        thermal_multiscale = self.thermal_multiscale_conv1(thermal)
        thermal_multiscale_pooled = nn.MaxPool2d(kernel_size=2, stride=2)(thermal_multiscale)

        rgb = rgb + thermal
        skip1 = rgb

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size())  # (120, 160)
        thermal = self.encoder_thermal_maxpool(thermal)
        if verbose: print("thermal.size() after maxpool: ", thermal.size())  # (120, 160)

        ########      rgb2+thermal2       ##########
        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size())  # (120, 160)
        thermal = self.encoder_thermal_layer1(thermal)
        if verbose: print("thermal.size() after layer1: ", thermal.size())  # (120, 160)

        rgb = rgb + thermal + thermal_multiscale_pooled
        skip2 = rgb

        ########    rgb3+thermal3    #########
        thermal_multiscale = self.thermal_multiscale_conv2(thermal)
        thermal_multiscale_pooled = nn.MaxPool2d(kernel_size=2, stride=2)(thermal_multiscale)

        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size())  # (60, 80)
        thermal = self.encoder_thermal_layer2(thermal)
        if verbose: print("thermal.size() after layer2: ", thermal.size())  # (60, 80)

        rgb = rgb + thermal + thermal_multiscale_pooled
        skip3 = rgb
        ########     rgb4+thermal4        #########
        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size())  # (30, 40)
        thermal = self.encoder_thermal_layer3(thermal)
        if verbose: print("thermal.size() after layer3: ", thermal.size())  # (30, 40)
        #
        rgb_weight = torch.sigmoid(self.rgb_weight_conv(rgb))
        thermal_weight = torch.sigmoid(self.thermal_weight_conv(thermal))
        rgb_fusion = rgb * rgb_weight + thermal * (1 - rgb_weight)
        thermal_fusion = thermal * thermal_weight + rgb * (1 - thermal_weight)
        rgb_enhanced = rgb_fusion * thermal
        thermal_enhanced = thermal_fusion * rgb
        rgb = rgb_enhanced + rgb
        thermal = thermal_enhanced + thermal
        skip4 = rgb + thermal

        ################### rgb5 ######################################
        rgb = self.encoder_rgb_layer4(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size())  # (15, 20)
        thermal = self.encoder_thermal_layer4(thermal)
        if verbose: print("thermal.size() after layer4: ", thermal.size())  # (15, 20)
        #
        cross_modal_feat = self.cross_modal_fusion(torch.cat([rgb, thermal], dim=1))
        fuse = rgb + thermal + cross_modal_feat
        fuse = self.feature_enhance(fuse)
        skip5 = fuse

        # ######################################################################

        # decoder
        fuse = self.deconv5(fuse)
        fuse = fuse + skip4
        if verbose: print("fuse after deconv1: ", fuse.size())  # (30, 40)
        fuse = self.deconv4(fuse)
        fuse = fuse + skip3
        if verbose: print("fuse after deconv2: ", fuse.size())  # (60, 80)
        fuse = self.deconv3(fuse)
        fuse = fuse + skip2
        edge_mask = self.edge(fuse)
        fuse = fuse + edge_mask
        hint = fuse
        if verbose: print("fuse after deconv3: ", fuse.size())  # (120, 160)
        fuse = self.deconv2(fuse)
        skip1 = self.skip_tranform(skip1)
        fuse = fuse + skip1
        if verbose: print("fuse after deconv4: ", fuse.size())  # (240, 320)
        fuse = self.deconv1(fuse)
        if verbose: print("fuse after deconv5: ", fuse.size())  # (480, 640)


        if self.training_style == 'train_student':
            if self.training:
                return edge_mask, fuse  # rgb_1    # fuse
            else:
                return skip5.detach(), fuse.detach(), hint.detach()
        else:
            return edge_mask, fuse  # fuse


class FeatureEnhanceModule(nn.Module):
    """特征增强模块"""

    def __init__(self, channels):
        super(FeatureEnhanceModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 4, channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        att = self.attention(x)
        x = x * att
        x = self.bn(x)
        x += identity
        return self.relu(x)


class TransBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out

class Attention(nn.Module):
    """注意力机制模块"""
    def __init__(self, dim, fmap_size, heads=4, dim_head=128, rel_pos_emb=False):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.pos_emb = AbsPosEmb(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        sim += self.pos_emb(q)

        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        return out


class AbsPosEmb(nn.Module):
    def __init__(self, fmap_size, dim_head):
        super().__init__()
        scale = dim_head ** -0.5
        self.scale = scale
        self.height = nn.Parameter(torch.randn(fmap_size[0], dim_head) * scale)
        self.width = nn.Parameter(torch.randn(fmap_size[1], dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb) * self.scale
        return logits

class BottleBlock(nn.Module):
    def __init__(self, dim, fmap_size, dim_out, proj_factor, downsample, heads=4, dim_head=128, rel_pos_emb=False, activation=nn.ReLU()):
        super().__init__()

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)
            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        attention_dim = dim_out // proj_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, attention_dim, 1, bias=False),
            nn.BatchNorm2d(attention_dim),
            activation,
            Attention(
                dim=attention_dim,
                fmap_size=fmap_size,
                heads=heads,
                dim_head=dim_head,
                rel_pos_emb=rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(heads*dim_head),
            activation,
            nn.Conv2d(heads*dim_head, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        nn.init.zeros_(self.net[-1].weight)
        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)

class BottleStack(nn.Module):

    def __init__(self, dim, fmap_size, dim_out=2048, proj_factor=4, num_layers=3, heads=4, dim_head=128,
                 downsample=True, rel_pos_emb=False, activation=nn.ReLU()):
        super().__init__()
        self.dim = dim
        self.fmap_size = fmap_size

        layers = []
        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample
            layer_fmap_size = (fmap_size[0] // (2 if downsample and not is_first else 1),
                               fmap_size[1] // (2 if downsample and not is_first else 1))

            layers.append(BottleBlock(
                dim=dim,
                fmap_size=layer_fmap_size,
                dim_out=dim_out,
                proj_factor=proj_factor,
                heads=heads,
                dim_head=dim_head,
                downsample=layer_downsample,
                rel_pos_emb=rel_pos_emb,
                activation=activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f'特征图通道数{c}必须与初始化时的通道数{self.dim}匹配'
        assert h == self.fmap_size[0] and w == self.fmap_size[1], f'特征图尺寸必须与初始化时的fmap_size{self.fmap_size}匹配'
        return self.net(x)


class newEdgeSeg(nn.Module):
    def __init__(self, cin1):
        super(newEdgeSeg, self).__init__()

        self.skip2conv = nn.Conv2d(in_channels=cin1, out_channels=cin1, kernel_size=1, stride=1)
        self.skip2bn = nn.BatchNorm2d(cin1)
        self.skip2relu = nn.ReLU()

        self.cdcm = CDCM(in_channels=cin1, out_channels=cin1)
        self.csam = CSAM(channels=cin1)
        self.conv1x1 = nn.Conv2d(in_channels=cin1, out_channels=1, kernel_size=1)

    def forward(self, x2):
        x2 = self.skip2conv(x2)
        x2 = self.skip2bn(x2)
        x2 = self.skip2relu(x2)

        x = self.cdcm(x2)
        x = self.csam(x)
        x = self.conv1x1(x)
        x = torch.sigmoid(x)
        return x


class CDCM(nn.Module):
    """
    Compact Dilation Convolution based Module
    """

    def __init__(self, in_channels, out_channels):
        super(CDCM, self).__init__()

        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2_1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv2_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=7, padding=7, bias=False)
        self.conv2_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=9, padding=9, bias=False)
        self.conv2_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=11, padding=11, bias=False)
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        x = self.relu1(x)
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x4 = self.conv2_4(x)
        return x1 + x2 + x3 + x4


class CSAM(nn.Module):
    """
    Compact Spatial Attention Module
    """

    def __init__(self, channels):
        super(CSAM, self).__init__()

        mid_channels = 4
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        nn.init.constant_(self.conv1.bias, 0)

    def forward(self, x):
        y = self.relu1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sigmoid(y)

        return x * y


def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(1)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(1)
    rtf_net = model(7).cuda(1)
    input = torch.cat((rgb, thermal), dim=1)
    rtf_net(input)


if __name__ == '__main__':
    unit_test()