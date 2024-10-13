# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from timm.loss import SoftTargetCrossEntropy
from .point_features import (
    get_uncertain_point_coords_with_randomness,
    get_depth_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..language.loss import ql_multi_contrastive_loss, image_text_contrastive_loss_queue, vl_similarity, all_gather_grad
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, _max_by_axis
from ..utils import box_ops

# from image2html.visualizer import VL


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


def calculate_depth_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    # logits (B, 1, h, w)
    # assert logits.shape[1] == 1
    # pred_logits = logits.clone()
    #
    # sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    # sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    #
    # gradient_x = F.conv2d(pred_logits, sobel_x.unsqueeze(0).unsqueeze(0))
    # gradient_y = F.conv2d(pred_logits, sobel_y.unsqueeze(0).unsqueeze(0))
    #
    # gradient_x = torch.abs(gradient_x)
    # gradient_y = torch.abs(gradient_y)
    #
    # gradient_x = F.pad(gradient_x, (1, 1, 1, 1))
    # gradient_y = F.pad(gradient_y, (1, 1, 1, 1))
    #
    # depth_uncertainty = pred_logits + gradient_x + gradient_y

    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return gt_class_logits




class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, top_x_layers, losses,
                 num_points, oversample_ratio, importance_sample_ratio):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.top_x_layers = top_x_layers
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_labels(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if layer_id > self.top_x_layers['mask']:
            return {"loss_mask_ce_0": 0}

        if indices is None or len(targets) == 0:
            loss_ce = outputs['pred_logits'].sum() * 0.0
            losses = {"loss_mask_ce_0": loss_ce}
            return losses

        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].type(self.empty_weight.dtype) # (B, num_queries, cls)


        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        # if src_logits.shape[2] == self.num_classes+1:
        #     empty_weight = torch.ones(self.num_classes + 1).to(src_logits.device).type(self.empty_weight.dtype)
        #     empty_weight[-1] = self.eos_coef
        # else:
        #     empty_weight = torch.ones(self.num_classes + 1000 + 1).to(src_logits.device).type(self.empty_weight.dtype)
        #     empty_weight[self.num_classes] = self.eos_coef

        # weight for each class, shape: (num_classes + 1)
        empty_weight = torch.ones(src_logits.shape[2]).to(src_logits.device).type(self.empty_weight.dtype)
        empty_weight[-1] = self.eos_coef

        # (B, num_classes + 1, num_queries) / (B, num_queries) / (num_classes + 1)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
        losses = {"loss_mask_ce_0": loss_ce}
        return losses

    def loss_labels_openimage(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if layer_id > self.top_x_layers['mask']:
            return {"loss_openimage_ce_0": 0}

        assert "pred_captions" in outputs

        if indices is None or len(targets) == 0 or (len(targets) > 0 and len(targets[0]['labels']) == 0):
            loss_ce = outputs['pred_captions'].sum() * 0.0
            losses = {"loss_openimage_ce_0": loss_ce}
            return losses

        # compute i2t loss
        loss_openimage_ce = 0
        losses = {}
        for b in range(len(indices)):
            pred_logit = outputs["pred_logits"][b][indices[b][0]]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((torch.arange(len(indices[b][1])), indices[b][1])).tolist()
            gt_logit[select_idx] = 1
            loss_openimage_ce += torch.sum(-gt_logit * F.log_softmax(pred_logit, dim=-1), dim=-1).mean()
        loss_openimage_ce = loss_openimage_ce / len(indices)
        losses.update({"loss_openimage_ce_0": loss_openimage_ce})
        return losses

    def loss_itc(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['retrieval']:
            return {"loss_retrieval_decoder_0": 0}
        t_emb = torch.cat([x['caption_proj'] for x in targets], dim=0)  # (B, C)
        v_emb = outputs['pred_captions'][:,-1] # (B, C)
        loss_contrast = image_text_contrastive_loss_queue(v_emb, t_emb, extra['lang_encoder'], extra['training'])

        # compute query-token contrastive loss
        ttk_emb = torch.cat([x['caption_tokens'] for x in targets], dim=0)  # (B, text_len, C)
        ttk_mask = torch.cat([x['caption_mask'] for x in targets], dim=0).float()  # (B, text_len)
        ttk_mask = ttk_mask * torch.cumsum(ttk_mask, dim=1)
        vtk_emb = outputs['pred_captions'][:,:-1]  # (B, 100, C)
        keep = torch.cat([x['caption_mask'] for x in targets], dim=0).bool()  # (B, text_len)

        ttk_emb = ttk_emb / (ttk_emb.norm(dim=-1, keepdim=True) + 1e-7)  # (B, text_len, C)

        vtk_emb = vtk_emb / (vtk_emb.norm(dim=-1, keepdim=True) + 1e-7)  # (B, 100, C)

        # print(ttk_emb[keep].shape)  # (keep_num, C)

        logit_scale = extra['lang_encoder'].logit_scale.exp().clamp(max=100)

        # prepare gt
        # Contrastive Loss,
        gt = (torch.eye(vtk_emb.shape[0]).type_as(ttk_mask).unsqueeze(-1) * ttk_mask.unsqueeze(0).repeat(vtk_emb.shape[0], 1, 1))[:,keep].flatten(1)
        gt = gt / (gt.sum(1, keepdim=True) + 1e-7)  # (B, keep_num)
        # compute i2t loss  (B, 100, C) @ (keep_num, C)^T
        logits = logit_scale * (vtk_emb @ ttk_emb[keep].transpose(0, 1)).mean(1)  # (B, 100, keep_num) -> (B, keep_num)

        loss_contrast_fine_vt = SoftTargetCrossEntropy()(logits, gt)
        # loss_contrast_fine = loss_contrast_fine_vt # i2t only

        # compute t2i loss
        bs, nq, _ = vtk_emb.shape
        logits = logit_scale * (ttk_emb @ vtk_emb.flatten(0,1).transpose(0, 1)).reshape(bs,-1,bs,nq).mean(dim=-1)[keep,:]
        loss_contrast_fine_tv = SoftTargetCrossEntropy()(logits, gt.t())
        # compute loss
        loss_contrast_fine = (loss_contrast_fine_vt * 0.7 + loss_contrast_fine_tv * 0.3)

        losses = {"loss_retrieval_decoder_0": loss_contrast + loss_contrast_fine * 0.5}
        return losses

    def loss_captionings(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['captioning']:
            return {"loss_captioning_0": 0}

        pred_captions_gen = outputs['pred_vlp'][:, :-1] # (B, Text_len-1, C)
        token_embs = extra['token_embedding'].weight  # (vocab_size, C)
        # token_embs = (token_embs / token_embs.norm(dim=-1, keepdim=True) + 1e-7)
        # pred_captions_gen = (pred_captions_gen / pred_captions_gen.norm(dim=-1, keepdim=True) + 1e-7)
        pred_captions_gen = pred_captions_gen @ token_embs.t()  # (B, Text_len-1, vocab_size)

        # temperature = extra['lang_encoder'].logit_scale
        # logit_scale = temperature.exp().clamp(max=100)

        target_captions_gen = torch.cat([target['answer_tokenids'] for target in targets], 0)[:, 1:]  # (B, Text_len-1)

        # not all caption have the same length, so pad and erase when compute loss
        target_captions_gen_mask = torch.cat([target['answer_mask'] for target in targets], 0)[:, 1:]  # (B, Text_len-1)


        # loss_caption = F.cross_entropy(pred_captions_gen.transpose(1,2) * logit_scale, target_captions_gen, reduction='none')
        loss_caption = F.cross_entropy(pred_captions_gen.transpose(1,2), target_captions_gen, reduction='none')  # (B, Text_len-1)
        loss_caption = (loss_caption * target_captions_gen_mask).sum() / (target_captions_gen_mask.sum() + 1)
        losses = {"loss_captioning_0": loss_caption}
        return losses


    def loss_vqa(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['vqa']:
            return {"loss_vqa_0": 0}

        pred_vqa_gen = outputs['pred_vlp'][:, :-1]  # (B, Text_len-1, C)
        token_embs = extra['token_embedding'].weight  # (vocab_size, C)
        pred_vqa_gen = pred_vqa_gen @ token_embs.t()  # (B, Text_len-1, vocab_size)


        target_vqa_gen = torch.cat([target['answer_tokenids'] for target in targets], 0)[:, 1:]  # (B, Text_len-1)

        # not all caption have the same length, so pad and erase when compute loss
        target_vqa_gen_mask = torch.cat([target['answer_mask'] for target in targets], 0)[:, 1:]

        pred_answerable = outputs['answerable']  # (B, 1)
        target_answerable = torch.stack([target['answerable'] for target in targets], 0).unsqueeze(1)  # (B, 1)


        loss_vqa_gen = F.cross_entropy(pred_vqa_gen.transpose(1, 2), target_vqa_gen,
                                       reduction='none')  # (B, Text_len-1)
        loss_vqa_gen = (loss_vqa_gen * target_vqa_gen_mask).sum() / (target_vqa_gen_mask.sum() + 1)

        loss_answerable = F.binary_cross_entropy_with_logits(pred_answerable, target_answerable)
        loss_vqa = 0.7 * loss_vqa_gen + 0.3 * loss_answerable
        losses = {"loss_vqa_0": loss_vqa}
        return losses


    def loss_masks(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        if layer_id >= self.top_x_layers['mask']:
            return {"loss_mask_bce_0": 0, "loss_mask_dice_0": 0}

        assert "pred_masks" in outputs
        if indices is None or len(targets) == 0:
            loss = outputs['pred_masks'].sum() * 0.0
            losses = {"loss_mask_bce_0": loss, "loss_mask_dice_0": loss}
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_mask_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_ocr(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['ocr']:
            return {"loss_ocr_0": 0}

        pred_ocr_gen = outputs['pred_vlp'][:, :-1]  # (B, Text_len-1, C)
        token_embs = extra['token_embedding'].weight  # (vocab_size, C)
        restrict_ids = extra['ocr_restrict_ids']

        pred_ocr_gen = pred_ocr_gen @ token_embs.t()  # (B, Text_len-1, vocab_size)
        penalty_mask = torch.ones_like(pred_ocr_gen).bool()
        penalty_mask[:, :, restrict_ids] = False

        pred_ocr_gen.masked_fill_(penalty_mask, float('-200'))

        # temperature = extra['lang_encoder'].logit_scale
        # logit_scale = temperature.exp().clamp(max=100)

        target_ocr_gen = torch.cat([target['answer_tokenids'] for target in targets], 0)[:, 1:]  # (B, Text_len-1)

        # not all caption have the same length, so pad and erase when compute loss
        target_ocr_gen_mask = torch.cat([target['answer_mask'] for target in targets], 0)[:,
                                   1:]  # (B, Text_len-1)

        pred_ocr_gen = pred_ocr_gen.float()

        loss_ocr = F.cross_entropy(pred_ocr_gen.transpose(1, 2), target_ocr_gen,
                                       reduction='none')  # (B, Text_len-1)

        loss_ocr = (loss_ocr * target_ocr_gen_mask).sum() / (target_ocr_gen_mask.sum() + 1)
        losses = {"loss_ocr_0": loss_ocr}
        return losses


    # def loss_ocr(self, outputs, targets, indices, num_masks, layer_id, extra):
    #     if layer_id >= self.top_x_layers['ocr']:
    #         return {"loss_ocr_0": 0}
    #
    #     token_embs = extra['token_embedding'].weight  # (vocab_size, C)
    #     restrict_ids = [49406, 271, 272, 273, 274, 275, 276, 277, 278, 279,
    #                     280, 320, 321, 322, 323, 324, 325, 326, 327, 328,
    #                     329, 330, 331, 332, 333, 334, 335, 336, 337, 338,
    #                     339, 340, 341, 342, 343, 344, 345, 49407]
    #
    #     target_ocr_gen = torch.cat([target['answer_tokenids'] for target in targets], 0)[:, 1:]  # (B, Text_len-1)
    #
    #     # not all caption have the same length, so pad and erase when compute loss
    #     target_ocr_gen_mask = torch.cat([target['answer_mask'] for target in targets], 0)[:,
    #                           1:]  # (B, Text_len-1)
    #     all_loss = []
    #     for i in range(len(outputs['pred_vlp'])):
    #         pred_ocr = outputs['pred_vlp'][i][:, :-1]  # (B, Text_len-1, C)
    #         pred_ocr_gen = pred_ocr @ token_embs.t()  # (B, Text_len-1, vocab_size)
    #
    #         penalty_mask = torch.zeros_like(pred_ocr_gen)
    #         penalty_mask[:, :, restrict_ids] = 1
    #         preferred_pred_ocr_gen = pred_ocr_gen * penalty_mask
    #
    #
    #         # temperature = extra['lang_encoder'].logit_scale
    #         # logit_scale = temperature.exp().clamp(max=100)
    #
    #         # pred_ocr_gen = pred_ocr_gen.float()
    #         preferred_pred_ocr_gen = preferred_pred_ocr_gen.float()
    #         # loss_ocr = F.cross_entropy(pred_ocr_gen.transpose(1, 2), target_ocr_gen,
    #         #                                reduction='none')  # (B, Text_len-1)
    #         penalty_loss_ocr = F.cross_entropy(preferred_pred_ocr_gen.transpose(1, 2), target_ocr_gen,
    #                                        reduction='none')  # (B, Text_len-1)
    #         loss_ocr = penalty_loss_ocr
    #
    #         loss_ocr = (loss_ocr * target_ocr_gen_mask).sum() / (target_ocr_gen_mask.sum() + 1)
    #         all_loss.append(loss_ocr)
    #     all_loss = sum(all_loss) / len(all_loss)
    #     losses = {"loss_ocr_0": all_loss}
    #     return losses

    def loss_groundings(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_gmasks" in outputs
        assert "pred_gtexts" in outputs

        if layer_id >= self.top_x_layers['grounding']:
            return {"loss_grounding_bce_0": 0, "loss_grounding_dice_0": 0, "loss_grounding_ce_0": 0}

        masks = [t["grounding_masks"] for t in targets]
        if indices is None or None in masks:
            loss = outputs['pred_gmasks'].sum() * 0.0
            return {"loss_grounding_bce_0": loss, "loss_grounding_dice_0": loss, "loss_grounding_ce_0": loss}

        pred_logits = []
        for b in range(len(indices)):
            t_emb = targets[b]['grounding_class_embs']
            v_emb = outputs["pred_gtexts"][b]

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

            out_prob = vl_similarity(v_emb, t_emb, temperature=extra['lang_logit'])
            pred_logits += [out_prob]
        outputs['pred_logits'] = pred_logits

        indices = self.matcher(outputs, targets, mode='grounding', extra={'temperature':extra['lang_logit']})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_gmasks"]
        src_masks = src_masks[src_idx]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]
        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_grounding_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, len(src_masks)),
            "loss_grounding_dice_0": dice_loss_jit(point_logits, point_labels, len(src_masks)),
        }

        # compute query-token contrastive loss
        # ttk_emb = torch.cat([x['caption_tokens'] for x in targets], dim=0)
        # ttk_mask = torch.cat([x['caption_mask'] for x in targets], dim=0).float()
        # ttk_mask = ttk_mask * torch.cumsum(ttk_mask, dim=1)
        # vtk_emb = outputs['pred_captions'][:,:-1]
        # keep = torch.cat([x['caption_mask'] for x in targets], dim=0).bool()

        # ttk_emb = ttk_emb / (ttk_emb.norm(dim=-1, keepdim=True) + 1e-7)
        # vtk_emb = vtk_emb / (vtk_emb.norm(dim=-1, keepdim=True) + 1e-7)
        # logit_scale = extra['lang_encoder'].logit_scale.exp().clamp(max=100)

        # # prepare gt
        # gt = (torch.eye(vtk_emb.shape[0]).type_as(ttk_mask).unsqueeze(-1) * ttk_mask.unsqueeze(0).repeat(vtk_emb.shape[0], 1, 1))[:,keep].flatten(1)
        # gt = gt / (gt.sum(1, keepdim=True) + 1e-7)
        # # compute i2t loss
        # logits = logit_scale * (vtk_emb @ ttk_emb[keep].transpose(0, 1)).mean(1)
        # loss_contrast_fine_vt = SoftTargetCrossEntropy()(logits, gt)
        # # loss_contrast_fine = loss_contrast_fine_vt # i2t only

        # # compute t2i loss
        # bs, nq, _ = vtk_emb.shape
        # logits = logit_scale * (ttk_emb @ vtk_emb.flatten(0,1).transpose(0, 1)).reshape(bs,-1,bs,nq).mean(dim=-1)[keep,:]
        # loss_contrast_fine_tv = SoftTargetCrossEntropy()(logits, gt.t())
        # # compute loss
        # loss_contrast_fine = (loss_contrast_fine_vt * 0.7 + loss_contrast_fine_tv * 0.3)

        # compute t2i loss
        loss_grd_ce = 0
        for b in range(len(indices)):
            task = targets[b]['grounding_task']
            pred_logit = outputs["pred_logits"][b]
            gt_logit = torch.zeros_like(pred_logit)
            select_idx = torch.stack((indices[b][0], indices[b][1])).tolist()
            gt_logit[select_idx] = 1
            t_hash = torch.tensor(targets[b]['grounding_hash'], device=gt_logit.device)
            hash_table = torch.zeros((len(t_hash), len(t_hash)), device=gt_logit.device)
            for idx in range(0, len(hash_table)):
                hash_table[idx][t_hash==t_hash[idx]] = 1
            hash_table = hash_table / hash_table.sum(-1, keepdim=True)
            gt_logit = gt_logit @ hash_table
            loss_grd_ce += self.grounding_weight[task]*torch.sum(-gt_logit.t() * F.log_softmax(pred_logit.t(), dim=-1), dim=-1).mean()
        loss_grd_ce = loss_grd_ce / len(indices)
        losses.update({"loss_grounding_ce_0": loss_grd_ce})
        del src_masks
        del target_masks
        return losses

    def loss_spatials(self, outputs, targets, indices, num_masks, layer_id, extra):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_smasks" in outputs
        assert "pred_smaskembs" in outputs

        if layer_id >= self.top_x_layers['spatial']:
            loss = outputs['pred_smasks'].sum() * 0.0
            loss_grd_ce = outputs["pred_smasks"].sum() * 0.0
            return {"loss_spatial_bce_0": loss, "loss_spatial_dice_0": loss, "loss_spatial_ce_0": loss_grd_ce}

        gt_masks = [x['gt_spatial_masks'] for x in targets]
        # compute a keep index with batch size to avoid empty gt_masks
        stack_gt_mask = torch.cat(gt_masks)
        bs,_,_ = stack_gt_mask.shape
        stack_gt_mask = stack_gt_mask.view(bs,-1).sum(dim=-1)
        keep = stack_gt_mask > 0 # only keep sample contain positive mask

        if keep.sum() == 0:
            loss = outputs['pred_smasks'].sum() * 0.0
            loss_grd_ce = outputs["pred_smasks"].sum() * 0.0
            return {"loss_spatial_bce_0": loss, "loss_spatial_dice_0": loss, "loss_spatial_ce_0": loss_grd_ce}

        # mask embedding logits
        v_emb = outputs["pred_smaskembs"] # [bs, nq, 512]

        # pos mask
        s_emb = outputs["pred_pspatials"] # [bs, ns, 512]
        pred_logits = v_emb @ s_emb.transpose(1,2)
        outputs['pred_pos_logits'] = pred_logits # [bs, nq, 1]
        indices = self.matcher(outputs, targets, mode='spatial', extra={})
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # pos class loss
        pred_logit = torch.cat([o[:len(t['gt_spatial_masks'])] for o,t in zip(outputs["pred_pos_logits"].transpose(1,2), targets)])
        gt_logit = torch.zeros_like(pred_logit)
        gt_logit = gt_logit[keep]
        _src_idx = [torch.arange(keep.sum(), device=src_idx[0].device), src_idx[1][keep.cpu()]]
        gt_logit[_src_idx] = 1
        pred_logit = pred_logit[keep]
        loss_spa_ce_pos = torch.sum(-gt_logit * F.log_softmax(pred_logit, dim=-1), dim=-1).mean()

        # neg mask
        # s_emb = outputs["pred_nspatials"] # [bs, ns, 512]
        # neg_mask = (s_emb.sum(dim=list(range(1, len(s_emb.shape)))) != 0).float()[keep]
        # pred_logits = v_emb @ s_emb.transpose(1,2)
        # outputs['pred_neg_logits'] = pred_logits # [bs, nq, 1]
        # indices = self.matcher(outputs, targets, mode='spatial_pn', extra=extra)
        # src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
        # src_masks_neg = outputs["pred_smasks"][src_idx][keep]
        # src_masks_neg = src_masks_neg*(neg_mask[:,None,None])
        # src_masks_neg = src_masks_neg.clip(0) * (-1)

        # neg class loss
        # pred_logit = outputs["pred_neg_logits"]
        # gt_logit = torch.zeros_like(pred_logit)
        # gt_logit[src_idx] = 1
        # bs,_,ns = pred_logit[keep].shape
        # pred_logit = pred_logit[keep].transpose(1,2).view(bs*ns,-1)
        # gt_logit = gt_logit[keep].transpose(1,2).view(bs*ns,-1)
        # loss_spa_ce_neg = (torch.sum(-gt_logit * F.log_softmax(pred_logit, dim=-1), dim=-1)*neg_mask).sum() / (neg_mask.sum()+1e-6)

        # recompute a keep index with matched tgt
        stack_gt_mask = nn.utils.rnn.pad_sequence(gt_masks, padding_value=-1).transpose(0,1)[tgt_idx]
        bs,_,_ = stack_gt_mask.shape
        target_masks = stack_gt_mask
        stack_gt_mask = stack_gt_mask.view(bs,-1).sum(dim=-1)
        keep = stack_gt_mask > 0 # only keep sample contain positive mask
        src_masks_pos = outputs["pred_smasks"][src_idx][keep]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks = target_masks.to(src_masks_pos)
        target_masks = target_masks[keep]

        # mul = extra['spatial_query_mode'][keep]
        # src_masks_cur = src_masks_cur.clip(0) * mul[:,None,None]
        # src_masks_cur = src_masks_cur

        # if neg_mask[0] == 1:
        #     import cv2
        #     print(src_masks_pos.shape)
        #     print(src_masks_neg.shape)
        #     print(target_masks.shape)
        #     # import pdb; pdb.set_trace()
        #     v_pos_mask = (src_masks_pos[0].sigmoid() > 0.5).float().cpu().detach().numpy() * 255
        #     v_neg_mask = (_src_masks_neg[0].sigmoid() > 0.5).float().cpu().detach().numpy() * 255
        #     v_sum = ((src_masks_pos[0]-_src_masks_neg[0].clip(0)).sigmoid() > 0.5).float().cpu().detach().numpy() * 255
        #     v_gt = target_masks[0].float().cpu().detach().numpy() * 255

        #     cv2.imwrite('v_pos_mask.png', v_pos_mask)
        #     cv2.imwrite('v_neg_mask.png', v_neg_mask)
        #     cv2.imwrite('v_sum.png', v_sum)
        #     cv2.imwrite('v_gt.png', v_gt)
        #     import pdb; pdb.set_trace()

        # src_masks = (src_masks_pos + src_masks_neg)[:, None]
        src_masks = src_masks_pos[:, None]
        target_masks = target_masks[:, None]

        # debug visualization
        # with torch.no_grad():
        #     import cv2
        #     import numpy as np

        #     v_src_masks = (F.interpolate(src_masks, size=target_masks.shape[-2:], mode='bilinear', align_corners=False).sigmoid() > 0.5).float().cpu().numpy()[:,0] * 255
        #     v_target_masks = target_masks.float().cpu().numpy()[:,0] * 255
        #     v_masks = np.concatenate([v_src_masks, v_target_masks], axis=2)

        #     for i in range(len(src_masks)):
        #         v1 = v_src_masks[i]
        #         v2 = v_target_masks[i]
        #         v = np.concatenate([v1,v2], axis=1)
        #         cv2.imwrite('v{}.png'.format(i), v)
        #     import pdb; pdb.set_trace()

        # visualization
        # VL.step()
        # v_img = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()
        # VL.add_image(v_img[:,:,::-1])
        # candidate_masks = batched_inputs[0]['spatial_query']['rand_shape'].float().cpu().numpy()
        # gt_masks = batched_inputs[0]['spatial_query']['gt_masks'].float().cpu().numpy()
        # texts = ['cmask' for i in range(len(candidate_masks))]
        # VL.overlay_obj_mask_to_image(v_img[:,:,::-1], candidate_masks, texts)
        # texts = ['gmask' for i in range(len(candidate_masks))]
        # VL.overlay_obj_mask_to_image(v_img[:,:,::-1], gt_masks, texts)

        # import cv2
        # for i in range(len(src_masks)):
        #     visual_src_mask_cur = (src_masks_cur[i].sigmoid()>0.5).detach().float().cpu().numpy() * 255
        #     visual_src_mask_mem = (src_masks_mem[i].sigmoid()>0.5).detach().float().cpu().numpy() * 255
        #     visual_src_mask = (src_masks[i,0].sigmoid()>0.5).detach().float().cpu().numpy() * 255
        #     visual_target_mask = (target_masks[i,0].sigmoid()>0.5).detach().float().cpu().numpy() * 255

        #     cv2.imwrite('visual_src_mask_cur_{}_{}.png'.format(i, mul[i].item()), visual_src_mask_cur)
        #     cv2.imwrite('visual_src_mask_mem_{}_{}.png'.format(i, mul[i].item()), visual_src_mask_mem)
        #     cv2.imwrite('visual_src_mask_{}_{}.png'.format(i, mul[i].item()), visual_src_mask)
        #     cv2.imwrite('visual_target_mask_{}_{}.png'.format(i, mul[i].item()), visual_target_mask)
        # import pdb; pdb.set_trace()

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            ).type(src_masks.dtype)
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        num_masks = len(src_masks)
        losses = {
            "loss_spatial_bce_0": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_spatial_dice_0": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        # losses.update({"loss_spatial_ce_0": loss_spa_ce_pos + loss_spa_ce_neg})
        losses.update({"loss_spatial_ce_0": loss_spa_ce_pos})

        del src_masks
        del target_masks
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, layer_id, extra):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if layer_id >= self.top_x_layers['box']:
            return {"loss_bbox_0": 0, "loss_giou_0": 0}

        assert 'pred_boxes' in outputs

        if indices is None or len(targets) == 0:
            loss = outputs['pred_boxes'].sum() * 0.0
            losses = {"loss_bbox_0": loss, "loss_giou_0": loss}
            return losses

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"]
        src_boxes = src_boxes[src_idx].sigmoid()

        target_boxes = [t['boxes'] for t in targets]
        max_size = _max_by_axis([list(box.shape) for box in target_boxes])
        max_size = [len(target_boxes)] + max_size
        empty_boxes = torch.zeros(max_size).to(src_boxes.device)
        for idx, tar_box in enumerate(target_boxes):
            empty_boxes[idx,:tar_box.shape[0],:] = tar_box
        target_boxes = empty_boxes[tgt_idx]

        # target_isthings = [t['is_things'] for t in targets]
        # max_size = _max_by_axis([list(lab.shape) for lab in target_isthings])
        # max_size = [len(target_isthings)] + max_size
        # empty_lab = torch.zeros(max_size).to(src_boxes.device)

        # for idx, tar_thing in enumerate(target_isthings):
        #     empty_lab[idx,:tar_thing.shape[0]] = tar_thing
        # target_isthings = empty_lab[tgt_idx]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox_0'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou_0'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depth(self, outputs, targets, indices, num_masks, layer_id, extra):
        if layer_id >= self.top_x_layers['depth']:
            return {"loss_depth_0": 0}
        assert 'pred_depth' in outputs

        pred_depth = outputs['pred_depth'].squeeze(1)  # (B, H, W)
        target_depth = torch.stack([target['depth_map'] for target in targets], 0)  # (B, H, W)

        valid_mask = (target_depth > 1e-5).detach()
        diff_log = torch.log(target_depth[valid_mask]) - torch.log(pred_depth[valid_mask])
        loss_depth = torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)


        losses = {"loss_depth_0": loss_depth}


        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks, layer_id, extra):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            "vqa": self.loss_vqa,
            'captionings': self.loss_captionings,
            'ocr':  self.loss_ocr,
            'depth': self.loss_depth,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks, layer_id, extra)

    def forward(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs_without_aux.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def forward_vlp(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        num_masks = indices = None
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def forward_grounding(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        indices = [[] for i in range(len(targets))]

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["grounding_masks"]) for t in targets) + 1e-7
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def forward_openimage(self, outputs, targets, extra=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        neg_class_emb =  all_gather_grad(torch.cat([x['neg_class_emb'] for x in targets]))
        neg_hash = all_gather_grad(torch.cat([x['neg_hash'] for x in targets]))

        extra['neg_class_emb'] = neg_class_emb
        extra['neg_hash'] = neg_hash
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, pred_logits = self.matcher.openimage_forward(outputs_without_aux, targets, extra=extra)
        outputs['pred_logits'] = pred_logits

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=neg_class_emb.device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, 0, extra))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            # NOTE: we reverse the aux_outputs so that the first is the second last layer
            for i, aux_outputs in enumerate(outputs["aux_outputs"][::-1]):
                indices, pred_logits = self.matcher.openimage_forward(aux_outputs, targets, extra=extra)
                aux_outputs['pred_logits'] = pred_logits
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, (i+1), extra)
                    l_dict = {k.replace('_0', f"_{i+1}"): v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num_losses=5):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num_losses, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

