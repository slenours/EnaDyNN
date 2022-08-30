"""
	Dense-Gated Model
"""
from imports import nn, torch, cp, F, math, OrderedDict

def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function

class _ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    From  https://github.com/ai-med/squeeze_and_excitation
    """
    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(_ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.add_module('se_layer', _ChannelSELayer(bn_size * growth_rate)),
        self.drop_rate = drop_rate
        self.efficient = efficient
    def forward(self, *prev_features):  #bottleneck -> selayer -> single denselayer
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        se_features = self.se_layer(bottleneck_output)
        new_features = self.conv2(self.relu2(self.norm2(se_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            selayer = _ChannelSELayer(num_input_features + i * growth_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            #print('name is : {}'.format(name))
            #print('layer is : {}'.format(layer))
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

class DGNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                small_inputs=False, efficient=True):
        super(DGNet, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'
        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                           ceil_mode=False))
        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)
        #Final BatchNom
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
    def forward(self, x):
        features = self.features(x)
        #print(self.features)
        return features

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)
    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.convA( torch.cat([up_x, concat_with], dim=1)  ) )  )

class Decoder(nn.Module):
    def __init__(self, num_features=363, decoder_width = 1.0):
        super(Decoder, self).__init__()
        features = int(num_features * decoder_width)
        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(skip_input=features//1 + 363, output_features=features//2)
        self.up2 = UpSample(skip_input=features//2 + 342,  output_features=features//4)
        self.up3 = UpSample(skip_input=features//4 + 300,  output_features=features//8)
        self.up4 = UpSample(skip_input=features//8 + 216,  output_features=features//16)
        self.up5 = UpSample(skip_input=features //16 + 24, output_features=features//32)
        self.conv3 = nn.Conv2d(features//32, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4, x_block5 = features[1],features[5], features[7], features[9], features[11], features[12]
        x_d0 = self.conv2(F.relu(x_block5))
        #print(x_d0.shape, x_block5.shape)
        x_d1 = self.up1(x_d0, x_block4)
        #print(x_d1.shape, x_block4.shape)
        x_d2 = self.up2(x_d1, x_block3)
        #print(x_d2.shape, x_block3.shape)
        x_d3 = self.up3(x_d2, x_block2)
        #print(x_d3.shape, x_block2.shape)
        x_d4 = self.up4(x_d3, x_block1)
        #print(x_d4.shape, x_block1.shape)
        x_d5 = self.up5(x_d4, x_block0)
        #print(x_d5.shape, x_block0.shape)
        #up_block = F.interpolate(x_d5, size=[x_d5.size(2)*2, x_d5.size(3)*2], mode='bilinear', align_corners=True)
        #print(up_block.shape)
        return self.conv3(x_d5)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.original_model = DGNet()
    def forward(self, x):
        features = [x]
        for k, v in self.original_model.features._modules.items():
          features.append(v(features[-1]))
         # print(k)
        #for item in features:
        #  print(item.shape)
        #print("*******************")
        return features

class DGModel(nn.Module):
    def __init__(self):
        super(DGModel, self).__init__()
        #self.encoder = Encoder()
        #self.decoder = Decoder()
        self.features = DGNet()
        #self.flat = nn.Flatten()
        self.fc1 = nn.Linear(363, 64)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        #return self.decoder(self.encoder(x))
        en_features = self.features(x).squeeze()
        #print('The output of DGNet : {}'.format(en_features.shape))
        #x0 = self.flat(en_features)
        #print('The output of Flatten layer : {}'.format(x0.shape))
        x1 = self.fc1(en_features)
        #print('The output of first FC layer : {}'.format(x1.shape))
        x2 = self.drop(x1)
        #print('The output of dropout layer : {}'.format(x2.shape))
        x3 = self.fc2(x2)
        #print('The output of Second FC layer : {}'.format(x3.shape))
        return x3