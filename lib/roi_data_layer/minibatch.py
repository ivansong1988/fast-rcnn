# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

#准备RoIDataLayer层top数据, RoIDataLayer::forward->_get_next_minibatch()调用
def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    #默认值为BATCH_SIZE = 128
    #num_images值由cfg.TRAIN.IMS_PER_BATCH = 2 确定
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    # 128/2
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    # fg比例默认0.25
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds)#图像缩放完之后统一存储在blob里, blob的尺寸为max_W*max_H

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32) #宽度为5的空数组[], 这里主要用来指明blob的形状
    labels_blob = np.zeros((0), dtype=np.float32)  #[]数组
    bbox_targets_blob = np.zeros((0, 4 * num_classes), dtype=np.float32) #第二维固定的[]数组
    bbox_loss_blob = np.zeros(bbox_targets_blob.shape, dtype=np.float32)
    # all_overlaps = []
    for im_i in xrange(num_images):
        #只保留特定数量的前景/背景proposals
        #每个batch随机从该副图像的proposal中随机选取固定数量的前景/背景proposal
        #也就是对roidb[im_i]提炼出固定大小、包含特定信息的一个集合
        labels, overlaps, im_rois, bbox_targets, bbox_loss \
            = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image,
                           num_classes) #按比例从roidb(bathsize长度)中提取roi信息

        # Add to RoIs blob 
        # 需要仔细想想
        rois = _project_im_rois(im_rois, im_scales[im_i])#根据im_scales完成bboxes的缩放
        batch_ind = im_i * np.ones((rois.shape[0], 1))#同一幅图像的proposals具有相同的索引号(同图像编号)
        rois_blob_this_image = np.hstack((batch_ind, rois))#拼接在一起(水平方向增加), 因此rois_blob_this_image[:,0]就是batch_ind
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))#rois_blob就是M个rois_blob_this_image拼接在一起(垂直方向增加), 因此可以知道rois_blob里面每一个

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))
        bbox_targets_blob = np.vstack((bbox_targets_blob, bbox_targets))
        bbox_loss_blob = np.vstack((bbox_loss_blob, bbox_loss))
        # all_overlaps = np.hstack((all_overlaps, overlaps))

    # For debug visualizations
    # _vis_minibatch(im_blob, rois_blob, labels_blob, all_overlaps)

    blobs = {'data': im_blob, #图像个数   ->conv1_1
             'rois': rois_blob, #图像个数*每个图像采样数 ->roi_pool5
             'labels': labels_blob} #图像个数*每个图像采样数 ->loss_cls

    if cfg.TRAIN.BBOX_REG:
        blobs['bbox_targets'] = bbox_targets_blob #图像个数*每个图像采样数  ->loss_bbox
        blobs['bbox_loss_weights'] = bbox_loss_blob #图像个数*每个图像采样数 ->loss_bbox

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes'] #当labels == 0时为background, 因为类标号为0,1,...,classes (num_class = classes + 1)
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    '''
    在网路最开始的预处理阶段SolverWrapper.__init__->add_bbox_regression_targets()时
    对部分样本(前景, ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0])计算了bbox回归时需要的偏移量
    而在网路训练过程中, 实际进行bboxes回归的样本明显是上面所求集合的一个子集, 因此cfg.TRAIN.FG_THRESH参数可能会与cfg.TRAIN.BBOX_THRESH
    不同. 实际上, 如果BBOX_THRESH<BBOX_THRESH会造成获取的样本里有bbox_targets为空, 并不能有效进行bbox回归的样本, 所以个人感觉要保证
    BBOX_THRESH>BBOX_THRESH
    '''
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0: #这里是随机选取不重复的fg_rois_per_this_image个
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image,
                             replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image,
                             replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0 #0为背景, 前景为1~classes
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]

    #bbox_targets/bbox_loss_weights的shape与roidb['bbox_targets'][keep_inds, :]相同
    #即同样包含了前景与背景proposal
    #但只有前景的bbox_loss_weights/bbox_targets非零
    #bbox_targets中只包含4维的偏移矢量, 但是被扩展为 4 * num_classes 宽度, 只有其所属类别的4个位置有值
    bbox_targets, bbox_loss_weights = \
            _get_bbox_regression_labels(roidb['bbox_targets'][keep_inds, :],
                                        num_classes) #背景bbox的bbox_targets全部为零

    return labels, overlaps, rois, bbox_targets, bbox_loss_weights #只包含前景/背景proposal, bbox_loss_weights记录每一维偏移矢量的权重(为'1')

def _get_image_blob(roidb, scale_inds):#scale的索引
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']: #只对在get_training_roidb中进行了flip操作的样本有效(config.py中默认进行水平镜像, 数据集倍增), 
            im = im[:, ::-1, :] #bboxes已经flipped了, 因此只需要对图像flip
        target_size = cfg.TRAIN.SCALES[scale_inds[i]] #默认是cfg.TRAIN.SCALES = 600
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size, #对于任何数据集都采用相同的PIXEL_MEANS?, 完成缩放\减均值\统一存入blob(按max_W,max_H)
                                        cfg.TRAIN.MAX_SIZE) #lib/utils/blob.py
        im_scales.append(im_scale)#记录实际的缩放比例, fast-rcnn默认短边到固定尺寸, 但同时也会限制最长边长度
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims) #im_blob, 一个batch图像取max_W,maxH相同尺寸

    return blob, im_scales#缩放比例用来处理bbox?

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded. (1-shot表示法)

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_loss_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0] #只有前景的sample
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_loss_weights[ind, start:end] = [1., 1., 1., 1.]
    return bbox_targets, bbox_loss_weights

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()
