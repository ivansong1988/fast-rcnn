#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import caffe
import argparse
import pprint
import numpy as np
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)
    #imdb_name = voc_2007_trainval
    #lib/datasets/factory.py, 实际调用pascal_voc(trainval, 2007), 返回lmdb
    #get_lmdb()函数实际的执行语句:
    #  return __sets[name]()
    #__sets[name]存放的是一个lambda函数指针, 这里调用该lambda函数的无参数形式(因为已经指定了参数的默认值)实际调用类pascal_voc的构造函数, 
    # 因此返回值imdb是一个pascal_voc对象
    # pascal_voc对象的__init__函数中完成了:
    # 1) self._classe: 设置类别标签; self._class_to_ind： 设置标签与标号的对应关系
    # 2) self._image_index = self._load_image_set_index() 
    #       读取VOCdevkit/VOC2007/ImageSets/Main/trainval.txt (由imdb_name根据get_imdb函数解析到VOC2007及trainval.txt)
    #       trainval.txt -> self._image_index
    # 3) self._roidb_handler = self.selective_search_roidb   
    #       设置函数句柄selective_search_roidb, 用于读取本地存储的proposal, 同时计算proposals与GT数据的ROI, 返回roidb (list)
    #       因此fast-rcnn算法的数据准备与预处理都是在这个函数中完成的 [**************************]
    #       lmdb类中的属性函数roidb()在_roidb为"None"时会调用_roidb_handler(), 
    #       因此在第一次通过imdb.roidb[i]访问_roidb元素时调用selective_search_roidb完成实际的数据加载与转换
    # 4) 其他值设定及校验self.config, self._devkit_path, self._data_path等    
    imdb = get_imdb(args.imdb_name) #返回pascal_voc对象
    print 'Loaded dataset `{:s}` for training'.format(imdb.name) #imdb.name继承自基类class imdb

    #lib/fast_rcnn/train.py
    #数据通过imdb._roidb_handler完成加载主要的触发函数为
    # imdb.append_flipped_images() 或者
    # rdl_roidb.prepare_roidb(imdb)
    # 总之, 在get_training_roidb函数中通过调用imdb._roidb_handler()完成数据加载和准备(细节参看lmdb的属性)
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
