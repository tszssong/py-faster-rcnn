# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue

class RoIDataLayer(caffe.Layer):
    """Fast R-CNN data layer used for training."""

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            print 'shuffle roidb better'
            fg_inds_bool = np.array([np.any(r['flags'] == 1) for r in self._roidb])
            bg_inds_bool = np.array([np.all(r['flags'] == 0) for r in self._roidb])
            fg_inds = np.where(fg_inds_bool)[0]
            bg_inds = np.where(bg_inds_bool)[0]
            fg_perm = np.random.permutation(np.arange(fg_inds.shape[0]))
            bg_perm = np.random.permutation(np.arange(bg_inds.shape[0]))
            fg_random_inds = fg_inds[fg_perm]
            bg_random_inds = bg_inds[bg_perm]
            num_images = cfg.TRAIN.IMS_PER_BATCH
            fg_ratio = float(len(fg_random_inds)) / (len(fg_random_inds) + len(bg_random_inds))
            fg_per_batch = int(fg_ratio * num_images)
            if fg_per_batch == 0:
                fg_per_batch = 1
            bg_per_batch = num_images - fg_per_batch
            assert fg_per_batch >0
            assert bg_per_batch >= 0
            print 'FG NUM ', fg_per_batch, 'BG NUM ', bg_per_batch
            inds = np.zeros((len(self._roidb)), dtype=np.int32)
            cur_fg = 0
            cur_bg = 0
            cur_inds = 0
            total_assign = len(inds) / num_images
            rest_assign = len(inds) % num_images
            for i in xrange(total_assign):
                if cur_fg + fg_per_batch > len(fg_random_inds):
                    cur_fg = 0
                if cur_bg + bg_per_batch > len(bg_random_inds):
                    cur_bg = 0
                inds[cur_inds:cur_inds + fg_per_batch] = fg_random_inds[cur_fg:cur_fg + fg_per_batch]
                inds[cur_inds + fg_per_batch: cur_inds + num_images] = bg_random_inds[cur_bg:cur_bg + bg_per_batch]
                cur_fg += fg_per_batch
                cur_bg += bg_per_batch
                cur_inds += num_images
            inds[cur_inds:cur_inds + rest_assign] = fg_random_inds[0:rest_assign]
            self._perm = inds
        else:
            print 'shuffle roidb'
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            return self._blob_queue.get()
        else:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            return get_minibatch(minibatch_db, self._num_classes)

    def _balance_roidb(self, roidb):
        class_dict = {}
        bg_inds = []
        r_index = 0
        for r in roidb:
            assert len(np.unique(r['flags'])) == 1
            if r['flags'][0] == 0:
                bg_inds.append(r_index)
                r_index += 1
                continue
            #assert len(np.unique(r['gt_classes'])) == 1
            if np.unique(r['gt_classes'])[0] not in class_dict:
                class_dict[np.unique(r['gt_classes'])[0]] = [r_index]
            else:
                class_dict[np.unique(r['gt_classes'])[0]].append(r_index)
            r_index += 1
        max_v = 0
        max_key = None
        for key in class_dict:
            print 'before', key, len(class_dict[key])
            if max_v < len(class_dict[key]):
                max_v = len(class_dict[key])
                max_key = key
        for key in class_dict:
            if key != max_key:
                add_num = max_v - len(class_dict[key])
                ori_len = len(class_dict[key])
                for _ in xrange(add_num):
                    class_dict[key].append(class_dict[key][np.random.randint(0, ori_len)])
        balance_roidb = []
        for key in class_dict:
            print 'after', key, len(class_dict[key])
            for roi_index in class_dict[key]:
                balance_roidb.append(roidb[roi_index])
        for ind in bg_inds:
            balance_roidb.append(roidb[ind])
        return balance_roidb

    def set_roidb(self, roidb):
        """balance roidb"""
        print 'begin balance roidb...'
        #roidb = self._balance_roidb(roidb)
        print 'end balance roidb'
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        if cfg.TRAIN.USE_PREFETCH:
            self._blob_queue = Queue(15)
            self._prefetch_process = BlobFetcher(self._blob_queue,
                                                 self._roidb,
                                                 self._num_classes)
            self._prefetch_process.start()
            # Terminate the child process when the parent exists
            def cleanup():
                print 'Terminating BlobFetcher'
                self._prefetch_process.terminate()
                self._prefetch_process.join()
            import atexit
            atexit.register(cleanup)
        else:
            self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            top[idx].reshape(1, 1)
            self._name_to_top_map['flags'] = idx
            idx += 1
        else: # not using RPN
            # rois blob: holds R regions of interest, each is a 5-tuple
            # (n, x1, y1, x2, y2) specifying an image batch index n and a
            # rectangle (x1, y1, x2, y2)
            top[idx].reshape(1, 5)
            self._name_to_top_map['rois'] = idx
            idx += 1

            # labels blob: R categorical labels in [0, ..., K] for K foreground
            # classes plus background
            top[idx].reshape(1)
            self._name_to_top_map['labels'] = idx
            idx += 1

            if cfg.TRAIN.BBOX_REG:
                # bbox_targets blob: R bounding-box regression targets with 4
                # targets per class
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_targets'] = idx
                idx += 1

                # bbox_inside_weights blob: At most 4 targets per roi are active;
                # thisbinary vector sepcifies the subset of active targets
                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_inside_weights'] = idx
                idx += 1

                top[idx].reshape(1, self._num_classes * 4)
                self._name_to_top_map['bbox_outside_weights'] = idx
                idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

class BlobFetcher(Process):
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, queue, roidb, num_classes):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._roidb = roidb
        self._num_classes = num_classes
        self._perm = None
        self._cur = 0
        self._shuffle_roidb_inds()
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED)

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            print 'shuffle roidb better'
            fg_inds_bool = np.array([np.any(r['flags'] == 1) for r in self._roidb])
            bg_inds_bool = np.array([np.all(r['flags'] == 0) for r in self._roidb])
            fg_inds = np.where(fg_inds_bool)[0]
            bg_inds = np.where(bg_inds_bool)[0]
            fg_perm = np.random.permutation(np.arange(fg_inds.shape[0]))
            bg_perm = np.random.permutation(np.arange(bg_inds.shape[0]))
            fg_random_inds = fg_inds[fg_perm]
            bg_random_inds = bg_inds[bg_perm]
            num_images = cfg.TRAIN.IMS_PER_BATCH
            fg_ratio = float(len(fg_random_inds)) / (len(fg_random_inds) + len(bg_random_inds))
            fg_per_batch = int(fg_ratio * num_images)
            if fg_per_batch == 0:
                fg_per_batch = 1
            bg_per_batch = num_images - fg_per_batch
            assert fg_per_batch >0
            assert bg_per_batch >= 0
            print 'FG NUM ', fg_per_batch, 'BG NUM ', bg_per_batch
            inds = np.zeros((len(self._roidb)), dtype=np.int32)
            cur_fg = 0
            cur_bg = 0
            cur_inds = 0
            total_assign = len(inds) / num_images
            rest_assign = len(inds) % num_images
            for i in xrange(total_assign):
                if cur_fg + fg_per_batch > len(fg_random_inds):
                    cur_fg = 0
                if cur_bg + bg_per_batch > len(bg_random_inds):
                    cur_bg = 0
                inds[cur_inds:cur_inds + fg_per_batch] = fg_random_inds[cur_fg:cur_fg + fg_per_batch]
                inds[cur_inds + fg_per_batch: cur_inds + num_images] = bg_random_inds[cur_bg:cur_bg + bg_per_batch]
                cur_fg += fg_per_batch
                cur_bg += bg_per_batch
                cur_inds += num_images
            inds[cur_inds:cur_inds + rest_assign] = fg_random_inds[0:rest_assign]
            self._perm = inds
        else:
            print 'shuffle roidb'
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        # TODO(rbg): remove duplicated code
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def run(self):
        print 'BlobFetcher started'
        while True:
            db_inds = self._get_next_minibatch_inds()
            minibatch_db = [self._roidb[i] for i in db_inds]
            blobs = get_minibatch(minibatch_db, self._num_classes)
            self._queue.put(blobs)
