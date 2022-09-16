"""
Build Branchy_DenseNet with 3 branches
Inspired by S. Teerapittayanon, B. McDanel, and H. Kung.
            Branchynet: Fast inference via early exiting from deep neural networks.
            https://gitlab.com/htkung/branchynet, 2016.
"""
from chainer import Chain, Variable
from chainer.backend import cuda
from branchynet.net import BranchyNet
from branchynet.links import *
#from net import BranchyNet
#from links import *
import chainer.functions as F
import chainer.links as L
import numpy as np
from six import moves

conv = lambda n: [L.Convolution2D(n, 32, 3, pad=1, stride=1), FL(F.relu)]  # conv
conv_2 = lambda n: [L.Convolution2D(n, 32, 1, pad=1, stride=1), FL(F.relu)]  # conv
cap = lambda n: [FL(F.max_pooling_2d, 3, 2), L.Linear(n, 10)]  # max pooling+fully connected


"""
class _DenseLayer(chainer.Chain):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False, ksize=1, stride=1):
        w = chainer.initializers.HeNormal()
        super(_DenseLayer, self).__init__(
            norm1=L.BatchNormalization(num_input_features),
            relu1=FL(F.relu),
            conv1=L.Convolution2D(num_input_features, bn_size * growth_rate,
                                           1, stride=1, pad=1, initialW=w),
            norm2=L.BatchNormalization(bn_size * growth_rate),
            relu2=FL(F.relu),
            conv2=L.Convolution2D(bn_size * growth_rate, growth_rate,
                                           3, stride=1, pad=1, initialW=w))
        self.drop_rate = drop_rate
        self.efficient = efficient
    '''
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.efficient = efficient
        self.ksize = ksize
        self.stride = stride

    def __deepcopy__(self, memo):
        new = type(self)(self.num_input_features, self.growth_rate, self.bn_size,
                         self.drop_rate, self.efficient, self.ksize, self.stride)
        return new
    '''

    def __call__(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, F.relu, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        #se_features = self.se_layer(bottleneck_output)
        new_features = self.conv2(F.relu(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(chainer.Chain):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False, ksize=1, stride=1):
        self.num_layers = num_layers
        super(_DenseBlock, self).__init__()

        for i in range(self.num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_link('denselayer%d' % (i + 1), layer)
            #print('denselayer%d' % (i + 1), layer)

    def __call__(self, init_features):
        features = [init_features]
        for layer in range(self.layers):
            new_features = layer(*features)
            features.append(new_features)
        print('features')
        print('concat', F.concat(features, axis=1))
        return F.concat(features, axis=1)


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = F.concat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function


class _ChannelSELayer(chainer.Chain):
    def __init__(self, num_channels, reduction_ratio=2):
        super(_ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = L.Linear(num_channels, num_channels_reduced, nobias=False)
        self.fc2 = L.Linear(num_channels_reduced, num_channels, nobias=False)

    def __call__(self, input_tensor):
        #print(input_tensor.shape)
        batch_size = input_tensor.shape[0]
        num_channels = input_tensor.shape[1]
        squeeze_tensor = np.reshape(input_tensor, (batch_size, num_channels, -1))
        #print('squeeze', squeeze_tensor)
        squeeze_tensor = np.mean(squeeze_tensor, axis=2)
        print('squeeze', squeeze_tensor)
        #squeeze_tensor = squeeze_tensor.astype(np.float32)
        #squeeze_tensor = F.cast(squeeze_tensor, np.float32)
        #print('squeeze', squeeze_tensor.shape)
        #print('22222', self.fc1(squeeze_tensor).shape)
        fc_out_1 = F.relu(self.fc1(squeeze_tensor))
        fc_out_2 = F.sigmoid(self.fc2(fc_out_1))
        a = squeeze_tensor.shape[0]
        b = squeeze_tensor.shape[1]
        output_tensor = np.multiply(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor
"""

class _DenseBlock(chainer.Chain):
    def __init__(self, in_ch, growth_rate, n_layer):
        self.n_layer = n_layer
        super(_DenseBlock, self).__init__()
        for i in moves.range(self.n_layer):
            '''
            self.add_link('bn1%d' % i,
                          L.BatchNormalization(in_ch))
            self.add_link('conv1%d' % i,
                          L.Convolution2D(in_ch, in_ch + i * growth_rate,
                                          1, 1, 1))
            '''
            #self.add_link('se_layer%d' % i, _ChannelSELayer(in_ch + i * growth_rate))
            self.add_link('bn%d' % (i + 1),
                          L.BatchNormalization(in_ch + i * growth_rate))
            self.add_link('conv%d' % (i + 1),
                          L.Convolution2D(in_ch + i * growth_rate, growth_rate,
                                          3, 1, 1))

    def __call__(self, x, dropout_ratio=0):
        for i in moves.range(1, self.n_layer+1):

            #se = self['se_layer%d' % (i-1)](x)
            #print(x)
            h2 = F.relu(self['bn%d' % i](x))
            #print(h2)
            h2 = F.dropout(self['conv%d' % i](h2), dropout_ratio)
            x = F.concat((x, h2))
        return x

class _Transition(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize=1, stride=1):
        w = chainer.initializers.HeNormal()
        super(_Transition, self).__init__(
            bn=L.BatchNormalization(in_channels),
            conv=L.Convolution2D(in_channels, out_channels, 1, stride=1, pad=1, initialW=w),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.stride = stride

    def __deepcopy__(self, memo):
        new = type(self)(self.in_channels, self.out_channels, self.ksize, self.stride)
        return new

    def __call__(self, x):

        h = F.average_pooling_2d(self.conv(F.relu(self.bn(x))), 2, 2)
        #print(h)
        return h

class DGNet:

    # build branches, n is the conv laysers of the first branch
    def build(self, n=1, percentTrainKeeps=1):
        if n == 0:
            branch1 = self.norm() + cap(12 * 4 * 4)
        else:
            branch1 = self.norm() + conv(12)
            for i in range(n - 1):
                branch1 += conv(32)
            branch1 += cap(32)

        # build the 2nd and the 3rd branches
        branch2 = self.norm() + conv_2(12) + cap(32)
        branch3 = self.norm() + conv_2(12) + cap(32)
        network = self.gen_2b(branch1, branch2, branch3)
        net = BranchyNet(network, percentTrianKeeps=percentTrainKeeps)

        return net

    # build BranchyDenseNet with 3 branches
    def gen_2b(self, branch1, branch2, branch3):
        network = [
            L.Convolution2D(3, 12, 7, pad=2, stride=3),
            L.BatchNormalization(12),
            FL(F.relu),
            FL(F.max_pooling_2d, 3, 1),
            _DenseBlock(12, 12, 16),
            _Transition(204, 12),
            Branch(branch1),
            #FL(L.Linear(12, 156)),
            #DenseBlock(156, 12, 12),
            #_Transition(12, 12),
            #FL(L.Linear(300, 12)),
            _DenseBlock(12, 12, 16),
            _Transition(204, 12),
            Branch(branch2),
            _DenseBlock(12, 12, 16),
            _Transition(204, 12),
            Branch(branch3),
            _DenseBlock(12, 12, 16),
            _Transition(204, 12),
            L.BatchNormalization(12),
            Branch([L.Linear(48, 10)])
        ]
        return network

    # ReLu->max pooling->norm
    def norm(self):
        Operation = [FL(F.relu), FL(F.max_pooling_2d, 3, 2), FL(
            F.local_response_normalization, n=3, alpha=5e-05, beta=0.75
        )]
        return Operation