# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg
import utils.cython_bbox

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    #imdb.roidb()是属性, 因此这条语句会执行roidb()函数, 内部调用imdb的_roidb_handler (= self.selective_search_roidb)完成roidb数据准备, 为imdb._roidb赋值
    roidb = imdb.roidb # gt与proposal叠放在一起
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i) #后处理, 为每一幅图像写入额外信息
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1) #确定属于哪个类别
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0) #刚好第0类最大时索引也为0
        # max overlap > 0 => class should not be zero (must be a fg class)#判断是否为背景论文中根据阈值
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)#异常值检测

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_targets'] = \
                _compute_targets(rois, max_overlaps, max_classes) #对于前景proposal, 记录类别标号, dx, dy ,dw, dh(R-CNN论文定义)
        ##'bbox_targets'大小与'boxes'相同, 但只有前景bboxes才有值

    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 4))
    squared_sums = np.zeros((num_classes, 4))
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 1:].sum(axis=0) #记录每个变化量的和
                squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0) #

    means = sums / class_counts   #这里记录的是dx ,dy, dw, dh各自的统计量, 针对每个类别
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    #对同一类的样本做标准化
    # Normalize targets
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_targets']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
            roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel() #返回array

'''准备用于bbox regression的数据：计算前景proposal box与gt的真实变换关系'''
## targets大小与roidb[im_i]['boxes']一样, 但只有前景bboxes才有值
def _compute_targets(rois, overlaps, labels):
    """Compute bounding-box regression targets for an image."""
    # Ensure ROIs are floats
    # roidb[im_i]['boxes']
    rois = rois.astype(np.float, copy=False)

    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0] #只有gt的overlap严格为1, proposal有overlap为1的概率极低
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0] #0.5, 只有大于该值的proposal才被作为正样本进行预测

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(rois[ex_inds, :],
                                                     rois[gt_inds, :]) #对每个高于阈值的proposal重新算一次与每个gt的重叠率ext_inds*gt_inds

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1) #len(ex_inds), 记录与每个前景proposal最匹配的gt box索引
    gt_rois = rois[gt_inds[gt_assignment], :] 
    ex_rois = rois[ex_inds, :]
    '''gt_rois与ex_rois配对, 存在多个proposal对应一个gt最佳匹配的情况'''
    
    '''box: [xmin, ymin, xmax, ymax]'''

   
    '''t_p - t_gt'''
    '''
    t_p : w*\phi_5: 预测值为回归网络输入的特征矢量的线性变换, 
                    需要学习的参数是w_{i}, i = 1,...,4, t_{i}分别对应dx(P), dy(P), dw(P), dh(P), 
                    均是P(预测, 也即输入特征矢量)的线性变换
    训练阶段的真实值就是预测的前景proposal box与GT boxes之间的变换, 由Eq(1)~(4)推出
    '''
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + cfg.EPS
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + cfg.EPS
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + cfg.EPS
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + cfg.EPS
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    '''eq(6)~eq(9)'''
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1] = targets_dx
    targets[ex_inds, 2] = targets_dy
    targets[ex_inds, 3] = targets_dw
    targets[ex_inds, 4] = targets_dh
    return targets
