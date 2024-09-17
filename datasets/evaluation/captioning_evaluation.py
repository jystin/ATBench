# Copyright (c) Facebook, Inc. and its affiliates.
# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import json
import logging
import itertools

import detectron2.utils.comm as comm
from detectron2.evaluation.evaluator import DatasetEvaluator

from caption_pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from vizwiz_api.vizwiz import VizWiz
from vizwiz_eval_cap.eval import VizWizEvalCap


class CaptioningEvaluator(DatasetEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).
    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        gt_json=None,
    ):
        self._logger = logging.getLogger(__name__)
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir
        if "coco" in self._dataset_name:
            self._gt_json = COCO(gt_json)
        else:
            self._gt_json = VizWiz(gt_json, ignore_rejected=True, ignore_precanned=True)

    def reset(self):
        self._gen_captions = []
        self._image_ids = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for output in outputs:
            self._image_ids.append(output['image_id'])
            self._gen_captions.append(output['captioning_text'])
        # print('------------------')
        # print(self._image_ids)
        # print(self._gen_captions)
        # print('------------------')
        # exit()

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        if self._distributed:
            comm.synchronize()
            def gather(x, move=False):
                x = comm.gather(x)
                x = list(itertools.chain(*x))
                if move:
                    x = [xx.to(self._gen_captions[0].device) for xx in x]
                return x
            gen_captions = gather(self._gen_captions)
            image_ids = gather(self._image_ids)
            if not comm.is_main_process():
                return {}
        else:
            gen_captions = self._gen_captions
            image_ids = self._image_ids

        assert len(gen_captions) == len(image_ids)
        pred_captions = [{"image_id": image_id, "caption": gen_caption} for image_id, gen_caption in zip(image_ids, gen_captions)]
        pred_pth = os.path.join(self._output_dir, 'results.json')
        json.dump(pred_captions, open(pred_pth, "w"))

        gt_captions = self._gt_json
        pred_captions = gt_captions.loadRes(pred_pth)

        # Evaluate the results
        if "coco" in self._dataset_name:
            cocoEval = COCOEvalCap(gt_captions, pred_captions)
            cocoEval.params['image_id'] = pred_captions.getImgIds()
            cocoEval.evaluate()
            result = cocoEval.eval
        else:
            vizwizEval = VizWizEvalCap(gt_captions, pred_captions)
            vizwizEval.params['image_id'] = pred_captions.getImgIds()
            vizwizEval.evaluate()
            result = vizwizEval.eval
        return result
