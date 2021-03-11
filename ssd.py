import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import data_cfg
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv1d for class conf scores
        2) conv1d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = data_cfg
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def feature_layer(cfg):
    layers = []
    for i in range(len(cfg)):
        layers.append([])
        for j in range(len(cfg[i]) - 1):
            layers[i].append(nn.Conv1d(cfg[i][j], cfg[i][j+1], kernel_size=3, stride = (2, 1)[i % 2== 1], padding =1))
    return layers

def extra_layer(cfg):
    layers = []
    for i in range(len(cfg) - 1):
        layers.append(nn.Conv1d(cfg[i], cfg[i+1], kernel_size = 3, stride = 2, padding = 1))
    return layers

def multibox(base_cfg, extra_cfg, num_box, num_classes):
    loc_layers = []
    conf_layers = []
    for i, channel in enumerate(base_cfg[-1][-2::]):
        loc_layers += [nn.Conv1d(channel,
                                 num_box[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv1d(channel,
                                  num_box[i] * num_classes, kernel_size=3, padding=1)]
    for i, channel in enumerate(extra_cfg, 2):
        loc_layers += [nn.Conv1d(channel,
                                 num_box[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv1d(channel,
                                  num_box[i] * num_classes, kernel_size=3, padding=1)]

    return loc_layers, conf_layers





def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)





if __name__ == '__main__':
    # 特征提取网络结构配置
    base_cfg = [[10, 32, 64, 128], [80, 128, 256, 128], [256, 128]]
    # extra网络结构配置
    extra_cfg = [128, 256, 256, 256, 512, 512]
    #各个网络对应的锚框数
    num_box = [4, 4, 4, 4, 6, 6, 4, 4]


    from data import Signal_Data
    dataset = Signal_Data(root = 'D:\signal_data\G35_recorder_npydata')
    iter_data = iter(dataset)
    data, label = next(iter_data)

    print(label[label.files[0]])

    class test_net(nn.Module):

        def __init__(self,base,extra_layer,cfg):
            super(test_net, self).__init__()
            self.feature_layer = []
            for i in range(len(cfg)):
                self.feature_layer.append(nn.ModuleList(base[i]))
            self.extra_layer = extra_layer
            self.cfg = cfg
        def forward(self, x):
            for i in range(len(self.cfg) - 1):
                for j in range(len(self.feature_layer[i])):
                    x[i] = x[i].to(torch.float32)
                    x[i] = self.feature_layer[i][j](x[i])
            result = torch.cat([x[0], x[1]], 1)
            for j in range(len(self.feature_layer[-1])):
                result = self.feature_layer[-1][j](result)
            for layer in self.extra_layer:
                result = layer(result)
            return result

    for i in range(len(data)):
        data[i] = data[i].unsqueeze(0)
    data = data[0:2]
    base = feature_layer(base_cfg)
    extra_layer = extra_layer(extra_cfg)
    box = multibox(base_cfg, extra_cfg, num_box, 21)


    net = test_net(base,extra_layer, base_cfg)
    result = net(data)

    print(result)





