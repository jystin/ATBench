# Modified by Xin Jiang from Xdecoder (https://arxiv.org/pdf/2212.11270.pdf)
from typing import Tuple
import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from timm.models.layers import trunc_normal_
from nltk.stem.lancaster import LancasterStemmer
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog

from .registry import register_model
from ..utils import configurable, get_class_names
from ..backbone import build_backbone, Backbone
from ..body import build_sem_seg_head
from ..modules import sem_seg_postprocess, SetCriterion, HungarianMatcher, AutomaticWeightedLoss, bbox_postprocess
from ..language import build_language_encoder
from ..language.loss import vl_similarity, image_text_contrastive_loss_queue
from ..language.misc import vl_similarity
from utils.prompt_engineering import prompt_engineering
from utils.constants import COCO_PANOPTIC_CLASSES

st = LancasterStemmer()


class ATModel(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        losses: dict,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        task_switch: dict,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        contxt_len: 77,
        test_topk_per_image: int,
        train_dataset_name: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.losses = losses
        self.num_queries = num_queries  # 101
        self.overlap_threshold = overlap_threshold  # 0.8
        self.object_mask_threshold = object_mask_threshold  # 0.8
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference  # True
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on

        # caption argument
        self.contxt_len = contxt_len
        self.task_switch = task_switch

        self.test_topk_per_image = test_topk_per_image
        if self.task_switch["mask"]:
            self.train_class_names = get_class_names(train_dataset_name)

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        dec_cfg = cfg['MODEL']['DECODER']

        # Loss parameters:
        # if compute aux loss
        deep_supervision = dec_cfg['DEEP_SUPERVISION']  # True
        # no object weight
        no_object_weight = dec_cfg['NO_OBJECT_WEIGHT']  # 0.1

        # loss weights, switcher for task, and top layers to compute loss
        loss_weights = {'mask': {'ce': dec_cfg['CLASS_WEIGHT'], 'dice': dec_cfg['DICE_WEIGHT'], 'bce': dec_cfg['MASK_WEIGHT']},
                        'captioning': dec_cfg['CAPTIONING_WEIGHT'],
                        'vqa': dec_cfg['VQA_WEIGHT'],
                        'depth': dec_cfg['DEPTH_WEIGHT'],
                        'ocr': dec_cfg['OCR_WEIGHT']}

        # 5 tasks
        task_switch = {'mask': dec_cfg.get('MASK', True),
                       'captioning': dec_cfg['CAPTIONING'].get('ENABLED', False),
                       'vqa': dec_cfg['VQA'].get('ENABLED', False),
                       'depth': dec_cfg['DEPTH'].get('ENABLED', False),
                       'ocr': dec_cfg['OCR'].get('ENABLED', False)}

        top_x_layers = {'mask': dec_cfg.get('TOP_MASK_LAYERS', 10),
                        'captioning': dec_cfg.get('TOP_CAPTIONING_LAYERS', 10),
                        'vqa': dec_cfg.get('TOP_VQA_LAYERS', 10),
                        'depth': dec_cfg.get('TOP_DEPTH_LAYERS', 10),
                        'ocr': dec_cfg.get('TOP_OCR_LAYERS', 10)}

        # build model
        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)  # focal_dw for image encode
        lang_encoder = build_language_encoder(cfg)  # vlpencoder text encoder
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape(), lang_encoder, extra)

        # building criterion
        matcher = HungarianMatcher(
            cost_class=loss_weights['mask']['ce'],
            cost_mask=loss_weights['mask']['bce'],
            cost_dice=loss_weights['mask']['dice'],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],
        )

        # init weight dict and criterion loss functions.
        losses = {'seg': [], 'cap': [], 'vqa': [], 'ocr': [], 'depth': []}
        if task_switch['mask']:
            losses['seg'] += ["labels", "masks"]
        if task_switch['depth']:
            losses['depth'] += ["depth"]
        if task_switch['ocr']:
            losses['ocr'] += ["ocr"]
        if task_switch['captioning']:
            losses['cap'] += ["captionings"]
        if task_switch['vqa']:
            losses['vqa'] += ["vqa"]


        weight_dict = {}
        # loss weight for last layer
        for key, turn_on in task_switch.items():
            if turn_on:
                if isinstance(loss_weights[key], dict):
                    # TODO HACK it should support bbox in the future
                    for key_, weight in loss_weights[key].items():
                        weight_dict["loss_{}_{}_0".format(key, key_)] = weight # NOTE: hard code for segmentation that has multiple loss
                else:
                    weight_dict["loss_{}_0".format(key)] = loss_weights[key]

        # generate full weight dict and remove not computed layers.
        if deep_supervision:
            dec_layers = dec_cfg['DEC_LAYERS']
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                for k, v in weight_dict.items():
                    if (i+1) > (top_x_layers[k.split('_')[1]] - 1):
                        continue
                    aux_weight_dict.update({k.replace('_0', f"_{i+1}"): v})
            weight_dict.update(aux_weight_dict)

        # generate critenrion for loss function.
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            top_x_layers=top_x_layers,
            eos_coef=no_object_weight,
            losses=[],
            num_points=dec_cfg['TRAIN_NUM_POINTS'],  # 12544 = 112*112
            oversample_ratio=dec_cfg['OVERSAMPLE_RATIO'],
            importance_sample_ratio=dec_cfg['IMPORTANCE_SAMPLE_RATIO'],
        )

        # extra logistic
        train_dataset_name = cfg['DATASETS']['TRAIN'][0] # HACK for only one training set.

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get(cfg['DATASETS']['TRAIN'][0]),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                or dec_cfg['TEST']['PANOPTIC_ON']
                or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "contxt_len": cfg['MODEL']['TEXT']['CONTEXT_LENGTH'],
            "test_topk_per_image": cfg['ADE20K']['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, mode=None):

        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """

        if self.training:
            losses = {}
            if self.task_switch['mask']:
                losses_seg = self.forward_seg(batched_inputs['seg'], task="seg")
                losses.update(losses_seg)
            if self.task_switch['captioning']:
                losses_cap = self.forward_vlp(batched_inputs['cap'], task="cap")
                losses.update(losses_cap)
            if self.task_switch['vqa']:
                losses_vqa = self.forward_vlp(batched_inputs['vqa'], task="vqa")
                losses.update(losses_vqa)
            if self.task_switch['ocr']:
                losses_ocr = self.forward_vlp(batched_inputs['ocr'], task="ocr")
                losses.update(losses_ocr)
            if self.task_switch['depth']:
                losses_depth = self.forward_depth(batched_inputs['depth'])
                losses.update(losses_depth)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else: # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            if mode == 'captioning':
                # auto-regressive
                return self.evaluate_vlp(batched_inputs, task="cap_infer")
            elif mode == 'vqa':
                return self.evaluate_vlp(batched_inputs, task="vqa_infer")
            elif mode == 'ocr':
                return self.evaluate_vlp(batched_inputs, task="ocr_infer")
            elif mode == 'depth':
                return self.evaluate_depth(batched_inputs)
            else:
                return self.evaluate(batched_inputs)


    def forward_seg(self, batched_inputs, task):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]  # List[image]
        images = ImageList.from_tensors(images, self.size_divisibility)  # pad

        # text embedding (num_clses, 512)
        self.sem_seg_head.predictor.lang_encoder.get_text_embeddings(self.train_class_names, is_eval=False)

        extra = {}
        # mask classification target
        # input bounding box is checked to be correct.
        targets = self.prepare_targets(batched_inputs, images)
        targets_vlp = self.prepare_vlp_targets(batched_inputs, images.tensor.device, answers=False)


        features = self.backbone(images.tensor)  # FPN multi-scale features
        outputs = self.sem_seg_head(features, target_vlp=targets_vlp, task=task, extra=extra)

        _outputs = {}
        for key, value in outputs.items():
            if key == 'pred_logits':
                _outputs[key] = value[:,:self.num_queries-1]
            elif key == 'pred_masks':
                _outputs[key] = value[:,:self.num_queries-1]
            elif key == 'aux_outputs':
                # The output of the middle layer
                _outputs[key] = []
                for i in range(len(value)):
                    _outputs[key] += [{}]
                    for _key, _value in value[i].items():
                        if _key == 'pred_logits':
                            _outputs[key][i][_key] = _value[:,:self.num_queries-1]
                        elif _key == 'pred_masks':
                            _outputs[key][i][_key] = _value[:,:self.num_queries-1]
        outputs = _outputs
        extra = {'lang_logit': self.sem_seg_head.predictor.lang_encoder.logit_scale,
                 'class_embeddings': getattr(self.sem_seg_head.predictor.lang_encoder, '{}_text_embeddings'.format('default'))}

        # bipartite matching-based loss
        self.criterion.losses = self.losses['seg'] # seg criterion losses
        losses = self.criterion(outputs, targets, extra)

        del outputs
        del _outputs
        return losses

    def forward_vlp(self, batched_inputs, task):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        targets_vlp = self.prepare_vlp_targets(batched_inputs, images.tensor.device)

        token_embedding = self.sem_seg_head.predictor.lang_encoder.lang_encoder.token_embedding

        ocr_restrict_ids = torch.tensor([49406, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                        280, 320, 321, 322, 323, 324, 325, 326, 327, 328,
                        329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
                        339, 340, 341, 342, 343, 344, 345, 49407], device=self.device)

        extra = {"token_embedding": token_embedding,
                 "lang_encoder": self.sem_seg_head.predictor.lang_encoder,
                 "training": self.training,
                 "ocr_restrict_ids": ocr_restrict_ids}

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=None, target_vlp=targets_vlp, task=task, extra=extra)

        for key, value in outputs.items():
            if key == 'pred_vlp':
                outputs[key] = value
            if task == "vqa" and key == 'answerable':
                outputs[key] = value
            elif key == 'aux_outputs':
                outputs[key] = []
                for i in range(len(value)):
                    outputs[key] += [{}]
                    for _key, _value in value[i].items():
                        if _key == 'pred_vlp':
                            outputs[key][i][_key] = _value
                        if task == "vqa" and _key == 'answerable':
                            outputs[key][i][_key] = _value

        if task == "cap":
            self.criterion.losses = self.losses['cap']
        if task == "vqa":
            self.criterion.losses = self.losses['vqa']
        if task == "ocr":
            self.criterion.losses = self.losses['ocr']
        losses = self.criterion.forward_vlp(outputs, targets_vlp, extra)
        del outputs
        return losses

    def forward_depth(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]  # List[image]
        images = ImageList.from_tensors(images, self.size_divisibility)  # pad

        depth_max_depth = batched_inputs[0]["depth_max_depth"]
        extra = {"depth_max_depth": depth_max_depth}
        targets = [{"depth_map": batch_per_image["depth_map"]} for batch_per_image in batched_inputs]

        targets_vlp = self.prepare_vlp_targets(batched_inputs, images.tensor.device, answers=False)

        features = self.backbone(images.tensor)  # FPN 
        outputs = self.sem_seg_head(features, target_vlp=targets_vlp, task="depth", extra=extra)

        _outputs = {}
        for key, value in outputs.items():
            if key == 'pred_depth':
                _outputs[key] = value
            elif key == 'aux_outputs':
                # The output of the middle layer
                _outputs[key] = []
                for i in range(len(value)):
                    _outputs[key] += [{}]
                    for _key, _value in value[i].items():
                        if _key == 'pred_depth':
                            _outputs[key][i][_key] = _value

        outputs = _outputs

        self.criterion.losses = self.losses['depth'] # seg criterion losses
        losses = self.criterion.forward_vlp(outputs, targets, extra)

        del outputs
        del _outputs
        return losses

    def evaluate(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        images = ImageList.from_tensors(images, self.size_divisibility)
        targets_vlp = self.prepare_vlp_targets(batched_inputs, images.tensor.device, answers=False)

        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_vlp=targets_vlp, target_queries=queries_grounding)

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        # upsample masks
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bicubic",
            align_corners=False,
            antialias=True
        )

        input_size = mask_pred_results.shape[-2:]
        keep_sem_bgd = self.metadata.keep_sem_bgd if hasattr(self.metadata, 'keep_sem_bgd') else False
        del outputs

        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)

            # semantic segmentation inference
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result, keep_sem_bgd)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r

            # panoptic segmentation inference
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r

            # instance segmentation inference
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, None)
                processed_results[-1]["instances"] = instance_r

        return processed_results

    def evaluate_depth(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]

        # images = ImageList.from_tensors(images, self.size_divisibility)
        images = ImageList.from_tensors(images)
        targets_vlp = self.prepare_vlp_targets(batched_inputs, images.tensor.device, answers=False)

        queries_grounding = None
        depth_max_depth = batched_inputs[0]["depth_max_depth"]
        extra = {"depth_max_depth": depth_max_depth}

        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features, target_queries=queries_grounding, target_vlp=targets_vlp, task="depth", extra=extra)

        depth_map_results = outputs["pred_depth"]

        # upsample depth map
        processed_results = []
        for depth_map_result in depth_map_results:
            processed_results.append({})
            processed_results[-1]["depth_map"] = depth_map_result.squeeze(0)

        return processed_results


    def evaluate_vlp(self, batched_inputs, task):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not hasattr(self, 'start_token'):
            self.start_token = torch.tensor([[49406]*self.contxt_len], device=self.device)

        ocr_restrict_ids = torch.tensor([49406, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                        280, 320, 321, 322, 323, 324, 325, 326, 327, 328,
                        329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
                        339, 340, 341, 342, 343, 344, 345, 49407], device=self.device)

        targets_vlp = self.prepare_vlp_targets(batched_inputs, images.tensor.device, answers=False)
        targets = targets_grounding = queries_grounding = None
        features = self.backbone(images.tensor)

        vlp_mask = None
        if 'vlp_mask' in batched_inputs[-1]:
            vlp_mask = torch.cat([x['vlp_mask'] for x in batched_inputs])

        outputs = self.sem_seg_head(features, target_queries=queries_grounding, task=task, target_vlp=targets_vlp,
                                    extra={'start_token': self.start_token,
                                           'vlp_mask': vlp_mask,
                                           'ocr_restrict_ids': ocr_restrict_ids,})

        processed_results = []
        for idx, batch_data in enumerate(batched_inputs):
            processed_results.append({})
            if task == "cap_infer":
                processed_results[-1]["captioning_token"] = outputs['pred_vlp'][idx]
                processed_results[-1]["captioning_text"] = outputs['pred_texts'][idx].split('.')[0]
                processed_results[-1]["image_id"] = batched_inputs[idx]['image_id']
            elif task == "ocr_infer":
                processed_results[-1]["ocr_token"] = outputs['pred_vlp'][idx]
                processed_results[-1]["ocr_text"] = outputs['pred_texts'][idx].split('.')[0]
                # processed_results[-1]["image_id"] = batched_inputs[idx]['image_id']
            elif task == "vqa_infer":
                processed_results[-1]["vqa_token"] = outputs['pred_vlp'][idx]
                processed_results[-1]["vqa_text"] = outputs['pred_texts'][idx].split('.')[0]
                processed_results[-1]["answerable"] = outputs['pred_answerable'][idx].item()
                processed_results[-1]["image_name"] = batched_inputs[idx]['image_name']
                processed_results[-1]["question_id"] = batched_inputs[idx]['question_id']
        return processed_results


    def prepare_vlp_targets(self, batched_inputs, device, answers=True):
        # encode Caption
        # encode Caption
        question_input_ids = []
        question_attention_mask = []
        if answers:
            answer_input_ids = []
            answer_attention_mask = []

        for cnt, x in enumerate(batched_inputs):
            question_input_ids += x['question_tokens']['input_ids'][:1]
            question_attention_mask += x['question_tokens']['attention_mask'][:1]

            if answers:
                answers = x['answers']
                randid = random.randint(0, len(answers) - 1)
                answer_input_ids += x['answer_tokens']['input_ids'][randid:randid+1]
                answer_attention_mask += x['answer_tokens']['attention_mask'][randid:randid+1]

        question_input_ids = torch.stack(question_input_ids)
        question_attention_mask = torch.stack(question_attention_mask)
        if answers:
            answer_input_ids = torch.stack(answer_input_ids)
            answer_attention_mask = torch.stack(answer_attention_mask)

        question_tokens = {"input_ids": question_input_ids, "attention_mask": question_attention_mask}
        question_lang_results = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(question_tokens,
                                                                                                   token=True)
        if answers:
            answer_tokens = {"input_ids": answer_input_ids, "attention_mask": answer_attention_mask}
            answer_lang_results = self.sem_seg_head.predictor.lang_encoder.get_text_token_embeddings(answer_tokens,
                                                                                                 token=True)

        target_vlp = []
        for cnt, x in enumerate(batched_inputs):
            target_dict = {}
            target_dict["question_tokens"] = question_lang_results['token_emb'][cnt:cnt + 1]  # (text_len, C)
            target_dict["question_proj"] = question_lang_results['class_emb'][cnt:cnt + 1]
            if answers:
                target_dict["answer_tokens"] = answer_lang_results['token_emb'][cnt:cnt + 1]  # (text_len, C)
                target_dict["answer_tokenids"] = answer_tokens['input_ids'][cnt:cnt + 1]
                target_dict["answer_mask"] = answer_tokens['attention_mask'][cnt:cnt + 1]
            if "answerable" in x:
                target_dict["answerable"] = x["answerable"]
            target_vlp.append(target_dict)
        return target_vlp

    def prepare_targets(self, batched_inputs, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for idx, batch_per_image in enumerate(batched_inputs):
            targets_per_image = batch_per_image["instances"].to(self.device)

            # pad gt_mask to image_shape
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            target_dict = {
                    "labels": targets_per_image.gt_classes,
                    "is_things": targets_per_image.is_things if targets_per_image.has("is_things") else None,
                    "masks": padded_masks,
                    # "boxes": gt_boxes
                    }

            new_targets.append(target_dict)

        return new_targets

    def semantic_inference(self, mask_cls, mask_pred, keep_sem_bgd=False):
        if keep_sem_bgd:
            mask_cls = F.softmax(mask_cls, dim=-1)
        else:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()  # qhw

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]   # keep, h, w
        cur_mask_cls = mask_cls[keep]  # keep, num_classes
        cur_mask_cls = cur_mask_cls[:, :-1] # keep, num_classes-1
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  
            stuff_memory_list = {}
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in thing_dataset_id_to_contiguous_id.values()

                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, box_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)

        labels_per_image = labels[topk_indices]
        topk_indices = (topk_indices // self.sem_seg_head.num_classes)
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]
        if box_pred is not None:
            box_pred = box_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            thing_dataset_id_to_contiguous_id = self.metadata.thing_dataset_id_to_contiguous_id if hasattr(self.metadata, 'thing_dataset_id_to_contiguous_id') else {}
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

            if box_pred is not None:
                box_pred = box_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)

        if box_pred is not None:
            result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        else:
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image

        return result



@register_model
def get_atmodel(cfg, **kwargs):
    return ATModel(cfg)