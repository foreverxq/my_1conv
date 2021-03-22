from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.

    为每个特征向量制作一个priorpos
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []

        ##原先图像的尺寸为300 * 300，将
        # 其用比例方式表示则，设输入图像为1 * 1， 则每个特征图所得到priorbox 都是相对于这个1* 1的比例坐标。
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
class prior_box(object):
    """
    为extra_layer所得到的特征向量制作初始锚框
    """
    def __init__(self, cfg):
        super(prior_box, self).__init__()
        self.signal_length = cfg['signal_length']
        self.num_priors = len(cfg['aspect_ratios'])
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for l in range(f):
                f_k = self.signal_length / self.steps[k]
                #signal center c
                c = l / f_k

                #aspect_ratio: 1
                #rel size: min_size
                s_k = self.min_sizes[k] / self.signal_length
                mean += [c, s_k]

                #aspect_ratio: 1
                #rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.signal_length))
                mean += [c, s_k_prime]
                #rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [c, s_k * sqrt(ar)]
                    mean += [c, s_k / sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 2)
        if self.clip:
            output.clamp_(max = 1, min = 0)
        return output












if __name__ == '__main__':

    cfg = {
        'num_classes': 21,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [110, 55, 28, 14, 7, 4],
        'signal_length': 1760,
        'steps': [16, 32, 64, 128, 256, 512],

        'min_sizes' : [176, 352, 651, 950,1249,1548],
        'max_sizes' : [352, 651,950,1249,1548, 1848],

        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'clip': True,
        'name': 'VOC',
    }

    prior_box = prior_box(cfg = cfg)
    output = prior_box.forward()
    print(prior_box)

