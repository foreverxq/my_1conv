# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD损失函数
    计算目标:
        1)对于目标每一个真实的bounding box 计算与其相匹配的所有prior box，计算方式就是通过 jaccard计算重叠面积。
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3)利用hard negative mining 技术过滤掉过多的负例框，从而使得正例与负例保持在1： 3.
    目标损失:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        其中Lconf是交叉熵损失，Lloc是L1损失。

        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number  of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.


    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data #目标的四个坐标。
            labels = targets[idx][:, -1].data #目标的类别
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)#loc_t shape (batch_size, num_priors, 4)
        conf_t = Variable(conf_t, requires_grad=False)#conf_t shape: torch.size(batch_size, num_priors)

        pos = conf_t > 0  #pos 找出模型 对 priors分类后 不为背景 的prior
        num_pos = pos.sum(dim=1, keepdim=True) # conf_t 中不为背景的个数

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)#loc_data shape: torch.size(batch_size,num_priors,4)  expand_as 扩充的维度上面的值为复制的原先维度上面的值。
        loc_p = loc_data[pos_idx].view(-1, 4)#pos_idx 筛选出prior框中 不为背景的框
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)#计算prior框与真实框的 smooth_l1_loss

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)  #conf_data shape: torch.size(batch_size,num_priors,num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1)) # batch_conf shape(batch_size * num_priors, num_classes)
                                                                                    #log_sum_exp 后shape 为(batch_size * num_priors, 1)
                                                                                    # conf_t shape(batch_size, num_priors)
                                                                                    #通过gather函数选取出模型对各个prior预测的值
                                                                                    #其shape为(batch_size * num_priors, 1)

        # Hard Negative Mining
        loss_c[pos.squeeze(0)] = 0  # filter out pos boxes for now                             loss_c shape: (batch_size, num_priors) #pos shape
        loss_c = loss_c.view(num, -1)#loss_c shape(batch_size, num_priors)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1) #获得loss_c下标所对应的损失值rank
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)#pos size(batch_size, num_priors) #num_neg shape == num_pos.shape
        neg = idx_rank < num_neg.expand_as(idx_rank) #使loss_c中损失在前num_neg位的下标为True


        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
#TODO 其中hard negative mining 写的很巧妙，再继续看下。