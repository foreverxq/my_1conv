from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")



train_cfg = {
    'train_root' : 'D:\signal_data\G35_recorder_npydata',
    'cuda' : True,

    'save_folder': 'D:\signal_data',
    'visdom': False,
    'min_dim' : 1024,
    'num_classes': 12,
}





if torch.cuda.is_available():
    if train_cfg.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not train_cfg.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(train_cfg['save_folder']):
    os.mkdir(train_cfg['save_folder'])


if torch.cuda.is_available():
    if train_cfg['cuda'] == True:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not train_cfg['cuda']:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')






def train():

    #训练数据集设置
    dataset = Signal_Data(root = train_cfg['train_root'],)

    if train_cfg.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', data_cfg['min_dim'], data_cfg['num_classes'])
    net = ssd_net

    if train_cfg.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if train_cfg.resume:
        print('Resuming training, loading {}...'.format(train_cfg.resume))
        ssd_net.load_weights(train_cfg.resume)
    else:
        vgg_weights = torch.load(train_cfg.save_folder + train_cfg.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if train_cfg.cuda:
        net = net.cuda()

    if not train_cfg.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)


    optimizer = optim.SGD(net.parameters(), lr=train_cfg.lr, momentum=train_cfg.momentum,
                          weight_decay=train_cfg.weight_decay)
    criterion = MultiBoxLoss(data_cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, train_cfg.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')


    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)
    step_index = 0

    # if args.visdom:
    #     vis_title = 'SSD.PyTorch on ' + dataset.name
    #     vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
    #     iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
    #     epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)




    # create batch iterator
    batch_iterator = iter(data_loader)
    images, targets = next(batch_iterator)


    for iteration in range(args.start_iter, cfg['max_iter']):
        # if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
        #     update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
        #                     'append', epoch_size)
            # reset epoch loss counters
        if iteration != 0 and (iteration % epoch_size == 0):
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]
        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()

        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.data.item()), end=' ')

        # if args.visdom:
        #     update_vis_plot(iteration, loss_l.data[0], loss_c.data[0],
        #                     iter_plot, epoch_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            print('Saving state, iter:', iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_COCO_' +
                       repr(iteration) + '.pth')
        torch.save(ssd_net.state_dict(),
                    args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


# def create_vis_plot(_xlabel, _ylabel, _title, _legend):
#     return viz.line(
#         X=torch.zeros((1,)).cpu(),
#         Y=torch.zeros((1, 3)).cpu(),
#         opts=dict(
#             xlabel=_xlabel,
#             ylabel=_ylabel,
#             title=_title,
#             legend=_legend
#         )
#     )

#
# def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
#                     epoch_size=1):
#     viz.line(
#         X=torch.ones((1, 3)).cpu() * iteration,
#         Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
#         win=window1,
#         update=update_type
#     )
#     # initialize epoch plot on first iteration
#     if iteration == 0:
#         viz.line(
#             X=torch.zeros((1, 3)).cpu(),
#             Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
#             win=window2,
#             update=True
#         )


if __name__ == '__main__':
    train()
    # cfg = voc
    # dataset = VOCDetection(root=args.dataset_root,
    #                        transform=SSDAugmentation(cfg['min_dim'],
    #                                                  MEANS))
    # data_loader = data.DataLoader(dataset, args.batch_size,
    #                               num_workers=args.num_workers,
    #                               shuffle=True, collate_fn=detection_collate,
    #                               pin_memory=True)
    # # create batch iterator
    # train()
    # batch_iterator = iter(data_loader)
    # test_images, test_targets = next(batch_iterator)
    # # for x in test_targets:
    # #     print(x)
    # print(np.ones(shape=(1, 5)))
    # tensor = torch.Tensor(np.ones(shape = (1, 5)))
    # ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    # net = ssd_net
    # out = net(test_images)
    # print(net(test_images))

