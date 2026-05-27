# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Q2L Transformer class.

Most borrow from DETR except:
    * remove self-attention by default.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
    * using modified multihead attention from nn_multiheadattention.py
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
from torch.nn import MultiheadAttention


import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_activation_fn
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from mmengine.model import BaseModule, ModuleList
from mmpretrain.structures import DataSample,label_to_onehot

from mmpretrain.registry import MODELS
from .Multi_label_RankCls_head import MultiLabelRankClsHead
from ..utils import LocallyConnected,DagmaMLP
from ..utils import SoftMoE,GroupedSoftMoE,GroupedLinear
from ..necks import MultiLabelSwAVNeck


@MODELS.register_module()
class Multilabel_BankCausal(MultiLabelRankClsHead):
    def __init__(self,position_embedding,img_size, patch_size=14, hidden_dim=512, nheads=8, enc_layers=6,
                 dec_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", pre_norm=False,
                 return_intermediate_dec=False, 
                 keep_other_self_attn_dec=True, keep_first_self_attn_dec=True,
                 swav_loss = None,
                 causal_type = 'implicit',
                 cals_pre='D-S',
                 with_global=False,
                 global_ratio=0.5,
                 confounder_norm=False,
                 **kwargs):
        """[summary]
    
        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__(**kwargs)
        self.cals_pre= cals_pre
        
        self.transformer =  Transformer(d_model=hidden_dim,
                                        dropout=dropout,
                                        nhead=nheads,
                                        dim_feedforward=dim_feedforward,
                                        num_encoder_layers=enc_layers,
                                        num_decoder_layers=dec_layers,
                                        normalize_before=pre_norm,
                                        return_intermediate_dec=return_intermediate_dec,
                                        rm_self_attn_dec=not keep_other_self_attn_dec, 
                                        rm_first_self_attn=not keep_first_self_attn_dec)
        self.num_class = self.num_classes
        self.causal_type = causal_type
        self.with_global = with_global
        self.diversity_weight = 0.05 
        self.global_ratio = global_ratio
        # self.confounder_norm = confounder_norm

        self.causal_Intervention = CausalIntervention(d_model=hidden_dim, num_heads=nheads,num_classes=self.num_classes,causal_type=self.causal_type,confounder_norm=confounder_norm)
        if self.cals_pre=='D-S':
            self.causal_fc = GroupWiseLinear(self.num_classes, hidden_dim, bias=True)
        else:
            self.causal_fc = nn.Linear(hidden_dim, self.num_classes)
        # self.input_proj = nn.Conv2d(self.in_channels, hidden_dim, kernel_size=1)
        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."
        
        # self.input_proj = nn.Conv2d(self.in_channels, hidden_dim, kernel_size=1)
        
            # self.Votingfc = GroupWiseLinear(self.num_classes, hidden_dim, bias=True)
        self.position_encoding = self.build_position_encoding(hidden_dim,position_embedding,img_size,patch_size)
        self.causal_position_encoding = PositionEmbeddingSine(hidden_dim//2, normalize=True, maxH=swav_loss.num_prototypes,maxW=1)
        self.fc = GroupWiseLinear(self.num_classes, hidden_dim, bias=True)
        # self.fc = CausalIntervention(d_model=hidden_dim, num_heads=nheads,num_classes=self.num_classes)
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim=hidden_dim
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        # self.position_embedding=position_embedding
        self.swav_loss = swav_loss
        if self.swav_loss:
            if not isinstance(swav_loss, nn.Module):
                swav_loss_module = MODELS.build(swav_loss)
            self.swav_loss_module = swav_loss_module
            self.swav_input_proj = MultiLabelSwAVNeck(self.in_channels,0,hidden_dim,with_avg_pool=True,with_l2norm=False)
            self.proto_diversity_lambda = 0.1  # 原型多样性正则系数
            # self.register_buffer('ema_prototypes', None)
            self.ema_prototypes = torch.zeros_like(self.swav_loss_module.prototypes,requires_grad=False)

        self.query_embed = nn.Embedding(self.num_classes, hidden_dim)
        # self.warm_up = warm_up

    def head_feature(self,features):
        features = self.pre_logits(features)
        out: List[Tensor] = []
        pos = []
        if isinstance(features, dict):
            for name, x in features.items():
                out.append(x)
                # position encoding
                pos.append(self.position_encoding(x).to(x.dtype))
                # pos.append(self[1](x).to(x.dtype))
        else:
            # for swin Transformer
            out.append(features)
            pos.append(self.position_encoding(features).to(features.dtype))
            # pos.append(self[1](features).to(features.dtype))

        # src, pos = self.backbone(input)
        src, pos = out[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        hs = self.transformer(src, query_input, pos) # B,K,d
        # out = hs[-1]
        return hs # return encoder result and decoder result
    def forward(self, features):
        # features = self.swav_input_proj(features) # grad_cam only
        hs = self.head_feature(features)[0][-1] # disease_specific feature
        # hs =nn.functional.normalize(hs, dim=-1, p=2)
      
        
        if self.swav_loss:    
            if self.current_epoch >= self.warmup_epoch:
                # if self.confounder_freezen:
                if self.swav_loss_module.with_sinkhorn:
                    if torch.all(self.ema_prototypes == 0):
                        # memory_bank =self.ema_prototypes
                        memory_bank = self.swav_loss_module.prototypes.clone().detach()
                    else:
                        memory_bank = self.ema_prototypes 
                else:
                    memory_bank = self.swav_loss_module.prototypes
                    # else:
                    #     memory_bank = self.swav_loss_module.prototypes
                    # memory_bank = F.normalize(memory_bank, dim=1, p=2)
                    # index = self.swav_input_proj.pooling(features[0])
                    # swav_loss = self.swav_loss_module(index)    
                    # memory_bank = self.swav_loss_module.prototypes.weight.data.clone()
                out = self.causal_Intervention(hs,memory_bank,self.swav_loss_module.prior)
                # print(self.swav_loss_module.prior)
                if self.current_epoch < self.warmup_epoch + 5:
                    base_out = self.causal_Intervention.out_proj(hs)
                    alpha = min(1.0, (self.current_epoch - self.warmup_epoch) / 5)
                    out = alpha * out + (1 - alpha) * base_out
                # out_2 = self.fc(hs1)
                # return out/2 +out_1/2
                # print(hs)
            else:
                # memory_bank = torch.zeros_like(self.swav_loss_module.prototypes,requires_grad=False)
                # out = self.causal_Intervention(hs,memory_bank,self.swav_loss_module.prior)
                out = self.causal_Intervention.out_proj(hs) 
                # out = self.fc(hs)
                # return out
        else:
            # out = self.fc(hs)
            out = self.causal_Intervention.out_proj(hs)
            # return out
        if self.with_global:
            out_1 = self.fc(hs)
            out = self.global_ratio * out_1 + (1 - self.global_ratio) * out
            return out
        else:
            return out
    
    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # cluster_loss = self._get_loss(cls_score, data_samples, **kwargs)
        feats = self.swav_input_proj(feats)
        head_feature= self.head_feature(feats)
        hs = head_feature[1]
        
        cls_score = self(feats)
        
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        if self.current_epoch >= self.warmup_epoch:
            losses['classification_loss'] = losses.pop('loss')
            # labels = torch.stack([i.gt_score.float() for i in data_samples]).repeat(int(cls_score.shape[0]/len(data_samples)),1)
            # diversity_loss = self.label_specific_diversity(head_feature[0][-1], labels)
            # current_weight = self.dynamic_diversity_weight()
            # current_weight = min(self.diversity_weight, 0.01 + 0.01 * (self.current_epoch - self.warmup_epoch)) 
            # losses['diversity_loss'] = current_weight * diversity_loss
        else:
             losses['classification_loss'] = 1.0* losses.pop('loss')
       
        if self.swav_loss and self.swav_loss_module.with_sinkhorn:
            # if self.current_epoch >= self.warmup_epoch:
            #     index = self.swav_input_proj.pooling(hs.detach())
            # else:
            #     index = self.swav_input_proj.pooling(hs)

            index = self.swav_input_proj.pooling(hs.detach())
            swav_loss = self.swav_loss_module(index) 
        # and self.swav_loss_module.prototypes.requires_grad and self.swav_loss_module.with_sinkhorn:
            # index = self.swav_input_proj.pooling(hs)
            # swav_loss = self.swav_loss_module(index) 

            # if self.swav_loss_module.prototypes.requires_grad and self.swav_loss_module.with_sinkhorn:
            # 原型多样性正则
            if self.swav_loss_module.prototypes.requires_grad:
                prototypes = self.swav_loss_module.prototypes
                with torch.no_grad():
                    sim_matrix = F.cosine_similarity(
                        prototypes.unsqueeze(1), 
                        prototypes.unsqueeze(0), 
                        dim=-1
                    )
                    mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
                    off_diag = sim_matrix[~mask]
                    diversity_loss = -off_diag.mean()
                swav_loss = swav_loss + self.proto_diversity_lambda * diversity_loss
                self.update_ema_prototypes(self.swav_loss_module.prototypes.data.clone())
            losses['swav_loss'] = swav_loss
 
        return losses
    def label_specific_diversity(self, hs, labels, var_lower_bound=0.01, margin=0.005):
        """
        多样性损失（惩罚类内方差过小），鼓励不同样本学到多样表达
        
        hs: [batch_size, num_classes, feature_dim]
        labels: [batch_size, num_classes]
        var_lower_bound: 方差下限（鼓励达到这个值）
        margin: 容差范围，防止轻微方差波动带来梯度震荡
        """
        diversity_loss = 0.0
        valid_labels = 0
        batch_size, num_classes, feat_dim = hs.shape

        for label_idx in range(num_classes):
            label_mask = labels[:, label_idx] > 0.5
            if label_mask.sum() > 1:
                label_features = hs[label_mask, label_idx, :]
                label_features = F.normalize(label_features, p=2, dim=1)

                feature_var = torch.var(label_features, dim=0)  # [feature_dim]

                # 只惩罚低于阈值的方差：ReLU形式
                penalty = F.relu(var_lower_bound - feature_var - margin)

                # 如果所有维度都高于阈值，loss为0
                loss_i = penalty.mean()

                if not torch.isnan(loss_i):
                    diversity_loss += loss_i
                    valid_labels += 1

        return diversity_loss / max(valid_labels, 1)
    
    def dynamic_diversity_weight(self):
        """动态调整多样性损失权重"""
        # 早期：较低权重
        if self.current_epoch < self.warmup_epoch + 10:
            return self.diversity_weight * 0.25
        # 中期：正常权重
        elif self.swav_loss_module.prototypes.requires_grad:
            return self.diversity_weight
        # 后期：稍高权重
        else:
            return self.diversity_weight * 1.5
    @torch.no_grad()
    def update_ema_prototypes(self, prototypes):
        """使用EMA更新原型"""
        if torch.all(self.ema_prototypes == 0):
            self.ema_prototypes = prototypes.clone()
        else:
            start_momentum = 0.1
            end_momentum = 0.9
            progress = min(self.current_epoch / max(self.warmup_epoch, 1), 1.0) 
            momentum = start_momentum + progress * (end_momentum - start_momentum)
            self.ema_prototypes.mul_(momentum).add_((1 - momentum) * prototypes)
            self.ema_prototypes = torch.nn.functional.normalize(self.ema_prototypes, p=2, dim=1)

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        num_classes = cls_score.size()[-1]
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            # target = torch.stack([i.gt_score.float().repeat(int(cls_score.shape[0]/len(data_samples))) for i in data_samples]).reshape(cls_score.shape[0], -1)
            # target = torch.stack([i.gt_score.float() for i in data_samples], dim=0)
            target = torch.stack([i.gt_score.float() for i in data_samples]).repeat(int(cls_score.shape[0]/len(data_samples)),1)
            
        else:
            target = torch.stack([
                label_to_onehot(i.gt_label, num_classes) for i in data_samples
            ]).float().repeat(int(cls_score.shape[0]/len(data_samples)),1)

        # compute loss
        losses = dict()
        # loss = self.loss_module(
        #     cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        # losses['loss'] = loss
        if self.Ranking:
            # if self.Voting:
            #     cls_score_ranking = cls_score -kwargs['Thr']
            # else:
            cls_score_ranking = cls_score
            Ranking_loss = self.log_sum_exp_pairwise_loss(cls_score_ranking,target)    
            # self.alpha.data.clamp_(min=0)
            losses['Ranking_loss']=self.alpha*Ranking_loss
        if self.Voting:
            # 'disease_count' in data_samples[0]:
            Decision_probility = cls_score -kwargs['Thr']
            kwargs.pop('Thr')
            # Decision_probility = self.decision_pairwise(cls_score, kwargs['Thr'])
            decision_loss =self.loss_module(Decision_probility, target, avg_factor=cls_score.size(0), **kwargs)
            # losses['Thr_loss'] = decision_loss
            losses['loss'] = decision_loss
        else:
            loss = self.loss_module(
            cls_score, target, avg_factor=cls_score.size(0), **kwargs)
            losses['loss'] = loss


        return losses
    
    def predict(self,
                feats: Tuple[torch.Tensor],
                data_samples: List[DataSample] = None,
                **kwargs) -> List[DataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[DataSample]: A list of data samples which contains the
            predicted results.
        """
        # The part can be traced by torch.fx
        # print(self.current_epoch)
        feats = self.swav_input_proj(feats)
        # if self.swav_loss and self.current_epoch >= self.warmup_epoch:
        #     cls_score,_ = self(feats)
        #     # print('i am using')
        # else:
        cls_score = self(feats)

       
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions
    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the logarithm of the MSE loss:
            .. math::
                \frac{d}{2} \log\left( \frac{1}{n} \sum_{i=1}^n (\mathrm{output}_i - \mathrm{target}_i)^2 \right)
        
        Parameters
        ----------
        output : torch.Tensor
            :math:`(n,d)` output of the model
        target : torch.Tensor
            :math:`(n,d)` input dataset

        Returns
        -------
        torch.Tensor
            A scalar value of the loss.
        """
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
    def causal_estimation(self,x):
        h_val = self.DagmaMLP.h_func(self.s)

        X_hat = self.DagmaMLP(x)
        score = self.log_mse_loss(X_hat, x)
        l1_reg = self.lambda1 * self.DagmaMLP.fc1_l1_reg(self.num_class)
        obj = self.mu * (score + l1_reg) + h_val
        return self.w* obj
    

    def build_position_encoding(self,hidden_dim, position_embedding,img_size,downsample_ratio=32):
        N_steps = hidden_dim // 2

        # if args.backbone in ['CvT_w24'] :
        #     downsample_ratio = 16
        # else:
        downsample_ratio = downsample_ratio

        if position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            assert img_size % downsample_ratio == 0, "args.img_size ({}) % 32 != 0".format(img_size)
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=img_size // downsample_ratio, maxW=img_size // downsample_ratio)
            # import ipdb; ipdb.set_trace()
        else:
            raise ValueError(f"not supported {position_embedding}")

        return position_embedding

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t,2,rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0),1,1,1))




class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x



class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, 
                 rm_self_attn_dec=True, rm_first_self_attn=True,
                 ):
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        if num_encoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.rm_self_attn_dec = rm_self_attn_dec
        self.rm_first_self_attn = rm_first_self_attn

        if self.rm_self_attn_dec or self.rm_first_self_attn:
            self.rm_self_attn_dec_func()

        # self.debug_mode = False
        # self.set_debug_mode(self.debug_mode)

    def rm_self_attn_dec_func(self):
        total_modifie_layer_num = 0
        rm_list = []
        for idx, layer in enumerate(self.decoder.layers):
            if idx == 0 and not self.rm_first_self_attn:
                continue
            if idx != 0 and not self.rm_self_attn_dec:
                continue
            
            layer.omit_selfattn = True
            del layer.self_attn
            del layer.dropout1
            del layer.norm1

            total_modifie_layer_num += 1
            rm_list.append(idx)
        # remove some self-attention layer
        # print("rm {} layer: {}".format(total_modifie_layer_num, rm_list))

    def set_debug_mode(self, status):
        print("set debug mode to {}!!!".format(status))
        self.debug_mode = status
        if hasattr(self, 'encoder'):
            for idx, layer in enumerate(self.encoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)
        if hasattr(self, 'decoder'):
            for idx, layer in enumerate(self.decoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed, mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if query_embed.ndim==2:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src

        tgt = torch.zeros_like(query_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        
        return hs.transpose(1, 2), memory[:h*w].permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
                            
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None
        self.omit_selfattn = False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        if not self.omit_selfattn:
            tgt2, sim_mat_1 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2, sim_mat_2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
                            
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class CausalIntervention(nn.Module):
    def __init__(self, d_model, num_heads,num_classes, causal_type='implicit',confounder_norm=False, dropout_p=0.1):
        """
        Args:
            d_model (int): Dimension of input features (X and C).
            num_heads (int): Number of attention heads.
            cluster_priors (torch.Tensor): Prior probabilities P(C=c) for each cluster.
        """
        super(CausalIntervention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_classes = num_classes
        self.causal_type = causal_type
        self.confounder_norm  = confounder_norm

        # Ensure the input dimension is divisible by the number of heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.context_norm = nn.LayerNorm(d_model)

        # Output projection
        # self.out_proj = nn.Linear(d_model, d_model)
        # self.out_proj = CausalGroupLinear(self.num_classes, self.num_heads, self.head_dim)
        self.out_proj = GroupWiseLinear(self.num_classes, self.d_model, bias=True)

        # Dropout
        self.attn_dropout = nn.Dropout(p=dropout_p)
        self.out_dropout = nn.Dropout(p=dropout_p)

        self.lambda_attn = nn.Parameter(torch.tensor(0.5))

        # Cluster priors (log probabilities)
        # self.register_buffer("cluster_priors", torch.log(cluster_priors))

    def forward(self, X, C, cluster_priors):
        """
        Args:
            X (torch.Tensor): Image features, shape (batch_size, d_model).
            C (torch.Tensor): Confounder cluster centroids, shape (num_clusters, d_model).
        Returns:
            torch.Tensor: Confounder-adjusted features, shape (batch_size, d_model).
        """
        batch_size, num_class, _ = X.size()
        num_clusters = C.size(0)
        if self.confounder_norm:
            X = F.normalize(X, dim=1, p=2)
        C = F.normalize(C, dim=1, p=2)

        # Project inputs to queries, keys, and values
        Q = self.query_proj(X)  # (batch_size, num_class, d_model)
        K = self.key_proj(C)    # (num_clusters, d_model)
        V = self.value_proj(C)  # (num_clusters, d_model)

         # Reshape for multi-head attention
        Q = Q.view(batch_size, num_class, self.num_heads, self.head_dim)  # (batch_size, num_class, num_heads, head_dim)
        K = K.view(num_clusters, self.num_heads, self.head_dim)  # (num_clusters, num_heads, head_dim)
        V = V.view(num_clusters, self.num_heads, self.head_dim)  # (num_clusters, num_heads, head_dim)

        # Compute attention scores
        scores = torch.einsum("bnhd,khd->bnkh", Q, K)  # (batch_size, num_class, num_clusters, num_heads)
        scores = scores / (self.head_dim ** 0.5)  # Scale by sqrt(head_dim)
        attn_weights = F.softmax(scores, dim=-2)
        # attn_weights = self.attn_dropout(attn_weights)
        if self.causal_type=='implicit':
            context = Q.unsqueeze(2) + V.unsqueeze(0).unsqueeze(0)
            # Add cluster priors as attention bias
            if cluster_priors is not None:
                cluster_priors = cluster_priors + 1e-8
                cluster_priors = cluster_priors / cluster_priors.sum()  # normalize
                prior_attn = cluster_priors.view(1, 1, -1, 1).to(scores.device)  # [1, 1, K, 1]
                λ = torch.clamp(self.lambda_attn, 0.0, 1.0) if isinstance(self.lambda_attn, torch.nn.Parameter) else self.lambda_attn
                # λ = torch.clamp(self.lambda_attn, 0.0, 1.0)
                # attn_weights = λ * attn_weights + (1 - λ) * prior_attn
                attn_weights = λ *attn_weights + (1 - λ) * prior_attn
                # attn_weights = attn_weights / attn_weights.sum(dim=2, keepdim=True) 
                # log_prior_scores = F.softmax(torch.log(cluster_priors),dim=-1) 
                # attn_weights = attn_weights + cluster_priors.unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(scores.device)  # (batch_size, num_class, num_heads, num_clusters)
            # attn_weights = F.softmax(attn_weights, dim=-2)
            # attn_weights = F.softmax(attn_weights, dim=-2)
            context = torch.einsum("bnkh,bnkhd->bnhd", attn_weights,context)
            context = context.view(batch_size, num_class,-1)
            context = self.context_norm(context)
            context = self.out_dropout(context)
            output =self.out_proj(context)
            # output = self.out_dropout(output)
        if self.causal_type=='explicit':
        # context = Q.unsqueeze(2) + V.unsqueeze(0).unsqueeze(0)
            # attn_weights = F.softmax(scores, dim=-2)
            ##----------type one attention(Q,K)* V +Q------
            context = torch.einsum("bnkh,khd->bnkhd", attn_weights,V)
            context = Q.unsqueeze(2) + context
            ##----------type two attention(Q,K)* (V+Q)---------
            # context = Q.unsqueeze(2) + V.unsqueeze(0).unsqueeze(0)
            # context = torch.einsum("bnkh,bnkhd->bnkhd", attn_weights,context)

            context = context.permute(0,2,1,3,4).reshape(batch_size*num_clusters,num_class,-1)
            context = self.context_norm(context)
            context = self.out_dropout(context)
            interoutput = self.out_proj(context).view(batch_size,num_clusters,num_class)
            # interoutput = self.out_dropout(interoutput).view(batch_size,num_clusters,num_class)
            # Compute attention weights
              # (batch_size, num_class, num_heads, num_clusters)

            # Apply attention to values
            # log_prior_scores = F.log_softmax(cluster_priors, dim=-1)
            # log_prior_scores = F.softmax(cluster_priors,dim=-1) 
            cluster_priors = cluster_priors + 1e-8
            cluster_priors = cluster_priors / cluster_priors.sum()
            output = torch.einsum("bkn,k->bn", interoutput, cluster_priors.to(interoutput.device))  # (batch_size, num_class, num_heads, head_dim)
            # context = context.reshape(batch_size, num_class, self.d_model)  # (batch_size, num_class, d_model)

            # Project output
            # output = self.out_proj(context)  # (batch_size, num_class, d_model)

        return output
class CausalGroupLinear(nn.Module):
    def __init__(self, num_class, num_heads, head_dim, bias=True):
        super().__init__()
        self.num_class = num_class  # K
        self.num_heads = num_heads  # D
        self.head_dim = head_dim
        self.bias = bias

        # Weight is (1, K, D) so all C share the same classifier
        self.W = nn.Parameter(torch.Tensor(1, num_class, num_heads, head_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class,1, num_heads))  # Bias shared across C

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(-1))  # Scaling by feature_dim (D)
        self.W.data.uniform_(-stdv, stdv)
        if self.bias:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: (B, K, C, D)
        # Expand W to match x for broadcasting: (1, K, 1, D)
        if x.dim()==3:
            B,L,D = x.shape
            x = x.view(B,L,self.num_heads, self.head_dim)
            x = (self.W * x).sum(dim=-1)
            if self.bias:# Bias broadcasting over (B, K, C)
                x = x + self.b.squeeze(2) 
            x = x.sum(-1) 
        else:
            B,C,S,H,D = x.shape
            x = (self.W.unsqueeze(2) * x).sum(dim=-1)  # Sum over D -> (B, K, C)

            if self.bias:
                x = x + self.b  # Bias broadcasting over (B, K, C)

        return x  # Output shape: (B, K, C)
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
        rm_self_attn_dec=not args.keep_other_self_attn_dec, 
        rm_first_self_attn=not args.keep_first_self_attn_dec,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
