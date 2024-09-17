# Modified by Xin Jiang from Xdecoder (https://arxiv.org/pdf/2212.11270.pdf)

import logging
from typing import Optional

import torch
torch.cuda.current_device()
from torch import nn, Tensor
from torch.nn import functional as F

from timm.models.layers import trunc_normal_
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init

from .registry import register_decoder
from .modules import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, PositionAttention
from ...utils import configurable, BeamHypotheses
from ...modules import PositionEmbeddingSine


class ATModelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        lang_encoder: nn.Module,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        dim_proj: int,
        num_queries: int,
        contxt_len: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        max_depth: float,
        task_switch: dict,
        vlp_step: int,
        enforce_input_project: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            lang_encoder: language encoder
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.contxt_len = contxt_len
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries  # 101
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.type_embed = nn.Embedding(2, hidden_dim)
        self.ques_pos_embed = nn.Embedding(contxt_len, hidden_dim)
        self.input_proj = nn.ModuleList()

        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        self.ques_proj =  nn.Parameter(torch.empty(dim_proj, hidden_dim))


        self.max_depth = max_depth
        self.task_switch = task_switch

        # output FFNs
        self.lang_encoder = lang_encoder
        if self.task_switch['mask']:
            self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.class_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
        trunc_normal_(self.class_embed, std=.02)

        # Caption Project and query
        if task_switch['captioning'] or task_switch['ocr'] or task_switch['vqa']:
            self.pos_embed_caping = nn.Embedding(contxt_len, hidden_dim)
            self.caping_embed = nn.Parameter(torch.empty(hidden_dim, dim_proj))
            trunc_normal_(self.caping_embed, std=.02)
            self.vlp_step = vlp_step

        if task_switch['ocr']:
            self.ocr_attention = PositionAttention(contxt_len, hidden_dim)

        if task_switch['vqa']:
            self.vqa_class_embed = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, 1),
            )

        if task_switch['depth']:
            self.depth_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        # register self_attn_mask to avoid information leakage, it includes interaction between object query, class query and caping query
        self_attn_mask = torch.zeros((1, num_queries + contxt_len, num_queries + contxt_len)).bool()
        self_attn_mask[:, :num_queries, num_queries:] = True # object+class query does not attend with caption query.
        self_attn_mask[:, num_queries:, num_queries:] = torch.triu(torch.ones((1, contxt_len, contxt_len)), diagonal=1).bool() # caption query only attend with previous token.
        self_attn_mask[:, :num_queries-1, num_queries-1:num_queries] = True # object query does not attend with class query.
        # self_attn_mask[:, num_queries-1:num_queries, :num_queries-1] = True # class query does not attend with object query.
        self.register_buffer("self_attn_mask", self_attn_mask)



    @classmethod
    def from_config(cls, cfg, in_channels, lang_encoder, mask_classification, extra):
        ret = {}

        ret["lang_encoder"] = lang_encoder
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        ret["hidden_dim"] = dec_cfg['HIDDEN_DIM']
        ret["dim_proj"] = cfg['MODEL']['DIM_PROJ']
        ret["num_queries"] = dec_cfg['NUM_OBJECT_QUERIES']
        ret["contxt_len"] = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']

        # Transformer parameters:
        ret["nheads"] = dec_cfg['NHEADS']
        ret["dim_feedforward"] = dec_cfg['DIM_FEEDFORWARD']

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert dec_cfg['DEC_LAYERS'] >= 1
        ret["dec_layers"] = dec_cfg['DEC_LAYERS'] - 1
        ret["pre_norm"] = dec_cfg['PRE_NORM']
        ret["enforce_input_project"] = dec_cfg['ENFORCE_INPUT_PROJ']
        ret["mask_dim"] = enc_cfg['MASK_DIM']
        ret["max_depth"] = dec_cfg['DEPTH'].get("MAX_DEPTH", 10.0)

        ret["task_switch"] = extra['task_switch']
        ret["vlp_step"] = dec_cfg.get('VLP_STEP', 50)
        return ret

    def forward(self, x, mask_features, mask=None, target_queries=None, target_vlp=None, task='seg', extra={}):
        if task in ["cap_infer", "ocr_infer", "vqa_infer"]:
            return self.forward_vlp(x, mask_features, mask=mask, target_queries=target_queries, target_vlp=target_vlp, task=task, extra=extra)

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask
        question_tokens = torch.cat([question['question_tokens'] for question in target_vlp], dim=0)  # (B, 77, 512)
        bs, _, _ = question_tokens.shape
        question_emb = ((question_tokens @ self.ques_proj) + self.type_embed.weight[1][None, None, :]).transpose(0, 1)  # (77, B, 512)

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:]) # /8 /16 /32
            pos.append(self.pe_layer(x[i], None).flatten(2))  # position embed
            # B, C, HW
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None] + self.type_embed.weight[0][None, :, None])  # proj + level embed

            # flatten NxCxHxW to HWxNxC
            pos[-1] = torch.cat([pos[-1].permute(2, 0, 1), self.ques_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)], dim=0)
            src[-1] = torch.cat([src[-1].permute(2, 0, 1), question_emb], dim=0)
        # QxNxC
        # pos embed
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # token feature embed
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_vlp = []
        predictions_answerable = []
        predictions_depth = []

        self_tgt_mask = None
        if self.training and (task == "cap" or task == 'ocr' or task == "vqa"):
            caping_lang_embed = torch.cat([caption['answer_tokens'] for caption in target_vlp], dim=0).transpose(0, 1) # language output
            _caping_lang_embed = caping_lang_embed.detach().clone()  # no grad copy
            output = torch.cat((output, _caping_lang_embed), dim=0) # (101 + text_len, B, C) concat object query, class token and caption token.
            caping_lang_embed += self.pos_embed_caping.weight.unsqueeze(1).repeat(1, bs, 1)  # (text_len, B, C)
            query_embed = torch.cat((query_embed, caping_lang_embed), dim=0) # (101 + text_len, B, C) may not add at the beginning.
            self_tgt_mask = self.self_attn_mask.repeat(output.shape[1]*self.num_heads, 1, 1)  # (B*h, 101 + text_len, 101 + text_len)

        else:
            self_tgt_mask = self.self_attn_mask[:,:self.num_queries,:self.num_queries].repeat(output.shape[1]*self.num_heads, 1, 1)

        # prediction heads on learnable query features (initial preparation work)
        results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], task=task, extra=extra)
        attn_mask = results["attn_mask"]
        predictions_class.append(results["outputs_class"])
        predictions_mask.append(results["outputs_mask"])
        predictions_vlp.append(results["outputs_vlp"])
        predictions_answerable.append(results["answerable"])
        predictions_depth.append(results["outputs_depth"])


        for i in range(self.num_layers):  # 9
            level_index = i % self.num_feature_levels
            # [B*h, Q, HW+context_len]
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            attn_mask = torch.cat((attn_mask, torch.zeros((attn_mask.shape[0], attn_mask.shape[1], self.contxt_len)).to(attn_mask.device)), dim=-1).bool()

            if self.training and (task == 'ocr' or task == 'cap' or task == 'vqa'):
                # [B*h, Q + contxt_len, HW + contxt_len]
                # For latent queries, we use a masked cross-attention mechanism as in [12], and full attention for the textual queries.
                attn_mask = torch.cat((attn_mask, torch.zeros_like(attn_mask[:, :self.contxt_len, :])), dim=1).bool()

            output, avg_attn = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=self_tgt_mask,  # mask interpretation paper
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )


            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], layer_id=i, task=task, extra=extra)
            attn_mask = results["attn_mask"]
            predictions_class.append(results["outputs_class"])
            predictions_mask.append(results["outputs_mask"])
            predictions_vlp.append(results["outputs_vlp"])
            predictions_answerable.append(results["answerable"])
            predictions_depth.append(results["outputs_depth"])


        assert len(predictions_class) == self.num_layers + 1
        if task == 'cap':
            out = {'pred_vlp': predictions_vlp[-1],   # (B, text_len, C)
                   'aux_outputs': [{'pred_vlp': x} for x in predictions_vlp[:-1]]}

            return out
        elif task == 'vqa':
            out = {'pred_vlp': predictions_vlp[-1],  # (B, text_len, C)
                   'answerable': predictions_answerable[-1],
                   'aux_outputs': [{'pred_vlp': x, 'answerable': y} for x, y in zip(predictions_vlp[:-1], predictions_answerable[:-1])]}
            return out
        elif task == "ocr":
            out = {'pred_vlp': predictions_vlp[-1],  # (B, text_len, C)
                   'aux_outputs': [{'pred_vlp': x} for x in predictions_vlp[:-1]]
                   }
            return out
        elif task == "depth":
            out = {'pred_depth': predictions_depth[-1],  # (B, 1, H, W)
                   'aux_outputs': [{'pred_depth': x} for x in predictions_depth[:-1]]
                   }
            return out
        else:
            out = {
                'pred_logits': predictions_class[-1],  # torch.Size([B, 101, num_classes])
                'pred_masks': predictions_mask[-1],  # torch.Size([B, 101, h, w])
                'aux_outputs': self._set_aux_loss(
                    predictions_class if self.mask_classification else None, predictions_mask
                )
            }
            return out

    def forward_vlp(self, x, mask_features, mask = None, target_queries = None, target_vlp = None, task='seg', extra={}):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask
        question_tokens = torch.cat([question['question_tokens'] for question in target_vlp], dim=0)  # (B, 77, 512)
        bs, _, _ = question_tokens.shape
        question_emb = ((question_tokens @ self.ques_proj) + self.type_embed.weight[1][None, None, :]).transpose(0, 1)  # (77, B, 512)

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))  # position embed
            # B, C, HW
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None] + self.type_embed.weight[0][None, :, None])  # proj + level embed

            # flatten NxCxHxW to HWxNxC
            pos[-1] = torch.cat([pos[-1].permute(2, 0, 1), self.ques_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)], dim=0)
            src[-1] = torch.cat([src[-1].permute(2, 0, 1), question_emb], dim=0)

        # QxNxC
        query_embed_ = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)
        answer_lang_token = extra['start_token'].repeat(bs, 1)  # (B, contxt_len)
        pos_embed_caping = self.pos_embed_caping.weight.unsqueeze(1).repeat(1, bs, 1)

        # prepare token embedding for evaluation
        token_embs = self.lang_encoder.lang_encoder.token_embedding.weight
        # token_embs = (token_embs / token_embs.norm(dim=-1, keepdim=True) + 1e-7)

        for cap_idx in range(0, self.vlp_step):
            answer_lang_embed = self.lang_encoder.forward_language_token((answer_lang_token,))[0].transpose(0, 1)
            output = torch.cat((query_feat, answer_lang_embed),
                               dim=0)  # concat object query, class token and caption token.
            answer_lang_embed += pos_embed_caping
            query_embed = torch.cat((query_embed_, answer_lang_embed), dim=0)  # may not add at the beginning.
            # output = torch.cat((query_feat, query_feat_caping), dim=0) # concat object query, class token and caption token.

            # prediction heads on learnable query features
            results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0],
                                                    task=task)
            attn_mask = results["attn_mask"]  # [B*h, Q, HW]

            restrict_ids = extra['ocr_restrict_ids']

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = torch.cat((attn_mask,
                                       torch.zeros((attn_mask.shape[0], attn_mask.shape[1], self.contxt_len)).to(
                                           attn_mask.device)), dim=-1).bool()
                attn_mask = torch.cat((attn_mask, torch.zeros_like(attn_mask[:, :self.contxt_len, :])), dim=1).bool()
                # output is going longer
                self_tgt_mask = self.self_attn_mask.repeat(output.shape[1] * self.num_heads, 1, 1)

                # mask some place to not calculate the attention
                if extra['vlp_mask'] is not None:
                    bs, nq, wh = attn_mask.shape
                    assert bs == self.num_heads, "Only support single image referring captioning."
                    cap_mask = extra['vlp_mask']
                    attn_mask = attn_mask.reshape(bs, nq, size_list[i % 3][0], size_list[i % 3][1])
                    cap_mask = F.interpolate(cap_mask[None,].float(), size_list[i % 3], mode='nearest').bool()[0, 0]
                    attn_mask[:, self.num_queries:, cap_mask] = True
                    attn_mask = attn_mask.reshape(bs, nq, wh)

                # attention: cross-attention first
                output, avg_attn = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=self_tgt_mask,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )

                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

                results = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[
                    (i + 1) % self.num_feature_levels], layer_id=i, task=task)
                attn_mask = results["attn_mask"]

            pred_vlp_gen = results['outputs_vlp']
            answerable = results['answerable']
            # pred_captions_gen = (pred_captions_gen / pred_captions_gen.norm(dim=-1, keepdim=True) + 1e-7)
            pred_vlp_gen = pred_vlp_gen @ token_embs.t()  # (B, contxt, C) @ (C, vocab_size)
            answer_lang_token[:, cap_idx + 1] = pred_vlp_gen[:, cap_idx].max(-1)[1]

            if task == "ocr_infer":
                answer_lang_token[:, cap_idx + 1] = pred_vlp_gen[:, cap_idx, restrict_ids].max(-1)[1]
                answer_lang_token[:, cap_idx + 1] = restrict_ids[answer_lang_token[:, cap_idx + 1]]
            else:
                answer_lang_token[:, cap_idx + 1] = pred_vlp_gen[:, cap_idx].max(-1)[1]

        texts = self.lang_encoder.tokenizer.batch_decode(answer_lang_token, skip_special_tokens=False)
        texts_new = []
        for x in texts:
            x = x.split('<|endoftext|>')[0]
            x = x.replace('<|endoftext|>', '')
            x = x.replace('<|startoftext|>', '')
            x = x.strip()
            if task == "ocr_infer":
                x = x.replace(' ', '')
            texts_new.append(x)

        out = {'pred_vlp': answer_lang_token,
               'pred_texts': texts_new,
               'pred_answerable': torch.sigmoid(answerable) if answerable is not None else None}
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, layer_id=-1, task='seg', extra=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)

        # extract image captioning token from decoder output
        if self.task_switch['captioning'] or self.task_switch['ocr'] or self.task_switch['vqa']:
            outputs_vlp = decoder_output[:,self.num_queries:] @ self.caping_embed
        else:
            outputs_vlp = None

        # recompute class token output. attention?
        norm_decoder_output = decoder_output / (decoder_output.norm(dim=-1, keepdim=True) + 1e-7)
        obj_token = norm_decoder_output[:, :self.num_queries - 1]
        cls_token = norm_decoder_output[:, self.num_queries - 1:self.num_queries]

        sim = (cls_token @ obj_token.transpose(1, 2)).softmax(-1)[:, 0, :,
              None]
        cls_token = (sim * decoder_output[:, :self.num_queries - 1]).sum(dim=1, keepdim=True)
        decoder_output = torch.cat((decoder_output[:, :self.num_queries - 1], cls_token), dim=1)

        # compute class, mask and bbox.
        class_embed = decoder_output @ self.class_embed

        # compute similarity between class_embed and dataset_text_embed
        if task=='seg':
            outputs_class = self.lang_encoder.compute_similarity(class_embed, fake=(((not self.task_switch['mask']) and self.training)))

        # Masked-attention mask
        if self.task_switch['mask'] and task=='seg':
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

            # For latent queries, we use a masked cross-attention mechanism as in [12], and full attention for the textual queries.
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bicubic", align_corners=False, antialias=True)

            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()

            # NOTE: fill False for cls token (JY)
            attn_mask[:, self.num_queries:self.num_queries+1].fill_(False)
            # attn_mask = torch.cat((attn_mask, torch.zeros_like(attn_mask[:, :, :self.contxt_len])), dim=-1)
        else:
            outputs_class = None
            outputs_mask = None
            attn_mask = torch.zeros((list(decoder_output.shape[:2]) + [attn_mask_target_size[0]*attn_mask_target_size[1]]), device=decoder_output.device).repeat(self.num_heads, 1, 1).bool()

        outputs_depth = None
        if self.task_switch['depth'] and task=='depth':
            depth_embed = self.depth_embed(cls_token)  # (B, 1, 512)
            depth_feature = torch.einsum("bqc,bchw->bqhw", depth_embed, mask_features)  # (B, 1, h, W)
            depth_feature = self.up(depth_feature)
            depth_feature = self.up(depth_feature)
            outputs_depth = torch.sigmoid(depth_feature) * extra['depth_max_depth'] + 1e-5

        answerable = None
        if self.task_switch['vqa'] and (task == 'vqa' or task == 'vqa_infer'):
            answerable = self.vqa_class_embed(class_embed[:, -1])  # class_embed: (B, 101, 512)

        results = {
            "outputs_class": outputs_class,
            "outputs_mask": outputs_mask,
            "attn_mask": attn_mask,  # [B*h, Q, HW]
            "answerable":  answerable,  # [B, 1]
            "outputs_vlp": outputs_vlp,
            "outputs_depth": outputs_depth,
        }
        return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


@register_decoder
def get_masked_transformer_decoder(cfg, in_channels, lang_encoder, mask_classification, extra):
    return ATModelDecoder(cfg, in_channels, lang_encoder, mask_classification, extra)
