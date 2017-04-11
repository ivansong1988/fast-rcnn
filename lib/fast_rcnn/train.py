# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
import roi_data_layer.roidb as rdl_roidb
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2

#实际训练的过程
class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        #在脚本里糅杂python层
        """Add information needed to train bounding-box regressors."""
        print 'Computing bounding-box regression targets...'
        #预先算好前景proposal及其对应gt的变换量, 用于bboxes回归, 其中变换量按类别进行了逐维标准化
        self.bbox_means, self.bbox_stds = \
                rdl_roidb.add_bbox_regression_targets(roidb)#lib/roi_data_layer/roidb.py
        print 'done'
        

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)
        
        # set_roidb: lib/roi_data_layer/layer.py::class RoIDataLayer
        # """Set the roidb to be used by this layer during training."""
        # self._roidb = roidb        
        # data_layer为python实现的RoIDataLayer
        # tops：
        #     data:   图像数据, 给第一个卷积层
        #     rois:   bboxes, 给ROI Pooling层
        #     labels: 给loss_cls, 分类网络损失层
        #     bbox_targets:  #给回归网络损失层
        #     bbox_loss_weights: #给回归网络损失层
        self.solver.net.layers[0].set_roidb(roidb) #data_layer

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if cfg.TRAIN.BBOX_REG:
            # save original values
            orig_0 = net.params['bbox_pred'][0].data.copy()
            orig_1 = net.params['bbox_pred'][1].data.copy()

            # scale and shift with bbox reg unnormalization; then save snapshot
            net.params['bbox_pred'][0].data[...] = \
                    (net.params['bbox_pred'][0].data *
                     self.bbox_stds[:, np.newaxis])
            net.params['bbox_pred'][1].data[...] = \
                    (net.params['bbox_pred'][1].data *
                     self.bbox_stds + self.bbox_means)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG:
            # restore net to original state
            net.params['bbox_pred'][0].data[...] = orig_0
            net.params['bbox_pred'][1].data[...] = orig_1

    #训练入口
    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()

            #不使用caffe的snapshot
            #参数设置从: 1)外部; 2)否则默认lib/fast-rcnn/config.py
            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        if last_snapshot_iter != self.solver.iter:
            self.snapshot()

def get_training_roidb(imdb): #imdb: pascal_voc类对象
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED: #这里的fip只对bbox做flip, 在lib/roi_data_layer/layer::RoIDataLayer层内部调用minibatch.py函数处理时再对图像flip
        '''
        也就是说如果需要对数据做flip, 就只能提前做好, fast-rcnn内部在RoiDataLayer对图像再做flip
        这里对所有图像做flip然后append到roidb
        roidb中每条记录：
        1)gt信息
        2)proposal信息
        3)是否翻转
        lib/fast_rcnn/config.py默认值__C.TRAIN.USE_FLIPPED = True #会做翻转, 因此数据量会翻倍, 这个操作是网络训练前最开始做的, 需要耗费时间
        '''
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images() #如果被调用, 会执行imdb.roidb_handler()完成数据准备,
        print 'done'

    print 'Preparing training data...'
    #lib/roi_data_layer/roidb.py
    #即使append_flipped_images()函数不被执行, 也会在prepare_roidb()函数中通过
    # roidb = imdb.roidb操作实际调用imdb.roidb_handler()完成加载(见lmdb::roidb()实现)
    rdl_roidb.prepare_roidb(imdb) #实现imdb._roidb赋值
    print 'done'

    return imdb.roidb #调用imdb::roidb(), 返回imdb._roidb

def train_net(solver_prototxt, roidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'
