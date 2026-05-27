# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.models import MultiLabelClsHead
from mmpretrain.structures import DataSample, label_to_onehot
# from .multi_label_vote_head import MultiLabelVoteHead

import torch.nn.functional as F
from ..utils import SoftMoE


@MODELS.register_module()
class MultiLabelRankClsHead(MultiLabelClsHead):
    """Linear classification head for multilabel task.

    Args:
        loss (dict): Config of classification loss. Defaults to
            dict(type='CrossEntropyLoss', use_sigmoid=True).
        thr (float, optional): Predictions with scores under the thresholds
            are considered as negative. Defaults to None.
        topk (int, optional): Predictions with the k-th highest scores are
            considered as positive. Defaults to None.
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).

    Notes:
        If both ``thr`` and ``topk`` are set, use ``thr` to determine
        positive predictions. If neither is set, use ``thr=0.5`` as
        default.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss: Dict = dict(type='CrossEntropyLoss', use_sigmoid=True),
                 Ranking = False, 
                 Voting = None,
                 average= False,
                 alpha = 1.0,
                 Ranking_weight = False,
                 cls_wise =True,
                 task: Optional[str] = 'multi-label', # 'multi-label' or 'binary-class'
                 thr: Optional[float] = None,
                 topk: Optional[int] = None,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01)):
        super(MultiLabelRankClsHead, self).__init__(
            loss=loss, thr=thr, topk=topk, init_cfg=init_cfg)

        assert num_classes > 0, f'num_classes ({num_classes}) must be a ' \
            'positive integer.'

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.Ranking = Ranking
        self.Voting = Voting
        self.average = average
        self.Ranking_weight = Ranking_weight
        self.cls_wise = cls_wise
        self.task = task
        if self.cls_wise:
            out_thr = num_classes
        else:
            out_thr = 1
        
        if Ranking:
            # self.alpha= nn.Parameter(torch.tensor(1.0, requires_grad=True))
            self.alpha = alpha
            if self.Ranking_weight:
                self.weight= nn.Parameter(torch.ones(num_classes))

        
        if Voting:
            if Voting =='smoe':
                 self.Votingfc = nn.Sequential(SoftMoE(in_features=in_channels,out_features=64, 
                                                  num_experts=self.num_classes,slots_per_expert=10),
                                          nn.ReLU(),
                                          SoftMoE(in_features=64,out_features=out_thr,num_experts=self.num_classes,slots_per_expert=10),
                                          ) 
            else:
                self.Votingfc = nn.Sequential(nn.Linear(in_channels, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, out_thr)
                                            )
           
            
            # SoftMoE(in_features=in_channels,out_features=self.num_classes,num_experts=self.num_classes,slots_per_expert=10)
            if average:
                self.gap = nn.AdaptiveAvgPool2d((1, 1))


        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The process before the final classification head.

        The input ``feats`` is a tuple of tensor, and each tensor is the
        feature of a backbone stage. In ``MultiLabelLinearClsHead``, we just
        obtain the feature of the last stage.
        """
        # The obtain the MultiLabelLinearClsHead doesn't have other module,
        # just return after unpacking.
        if isinstance(feats, (tuple, list)):
            feat = feats[-1]
        else:
            feat = feats
        return feat

    def forward(self, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        """The forward process."""
        pre_logits = self.pre_logits(feats)
        # if self.average:
        #     # batch = pre_logits.size(0)
        #     pre_logits = self.gap(pre_logits)
        #     pre_logits = pre_logits.view(pre_logits.size(0), -1)
            # pre_logits = outs
        # The final classification head.
        cls_score = self.fc(pre_logits)
        return cls_score
    
    def Votingforward(self,feats: Tuple[torch.Tensor]) -> torch.Tensor:
        if isinstance(feats, tuple):
            pre_logits = self.pre_logits(feats)
        else:
            pre_logits =feats

        if self.average:
                pre_logits = self.gap(pre_logits)
                pre_logits = pre_logits.view(pre_logits.size(0), -1)
        if self.Voting=='smoe':
            if pre_logits.ndim==2: #average pooling
            # The final classification head.
                pre_logits = pre_logits.unsqueeze(1)
            elif pre_logits.ndim==3:  #head_wise
                pre_logits = pre_logits.unsqueeze(2)
            elif pre_logits.ndim==4: # patch_wise threshold
                b,c,_,_=pre_logits.shape
                pre_logits = pre_logits.view(b,c,-1).permute(0,2,1)
                cls_score = self.Votingfc(pre_logits)
                cls_score= torch.mean(cls_score,dim=1)
                if not self.cls_wise:
                    cls_score = cls_score.repeat(1,self.num_classes)
                return cls_score
            cls_score = self.Votingfc(pre_logits)
            cls_score = cls_score.squeeze()
        else:
                # pre_logits = outs
            # cls_score = self.Votingfc(outs)
            cls_score = self.Votingfc(pre_logits)
        if not self.cls_wise:
            cls_score = cls_score.repeat(1,self.num_classes)
        return cls_score
        
 
    
    
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
        # The part can be traced by torch.fx
        cls_score = self(feats)
        # if self.Voting:
        #     Thr = self.Votingfc(feats)
        #     cls_score = self.decision_pairwise(cls_score,Thr)
        if self.Voting:
            Thr = self.Votingforward(feats)
            kwargs['Thr'] = Thr
        # predictions = self._get_predictions(cls_score, data_samples)
        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score[:len(data_samples),:], data_samples, **kwargs)
        # else:
        #     losses = self._get_loss(cls_score, data_samples)
        return losses
        

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[DataSample], **kwargs):
        """Unpack data samples and compute loss."""
        num_classes = cls_score.size()[-1]
        # Unpack data samples and pack targets
        if 'gt_score' in data_samples[0]:
            target = torch.stack([i.gt_score.float() for i in data_samples])
            # target = torch.stack([i.gt_score.float() for i in data_samples]).repeat(int(cls_score.shape[0]/len(data_samples)),1)
            # target = torch.stack([i.gt_score.float() for i in data_samples]).repeat(int(cls_score.shape[0]/len(data_samples)),1)
        else:
            target = torch.stack([
                label_to_onehot(i.gt_label, num_classes) for i in data_samples
            ]).float()
            # target = torch.stack([
            #     label_to_onehot(i.gt_label, num_classes) for i in data_samples
            # ]).float().repeat(int(cls_score.shape[0]/len(data_samples)),1)

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
    
    def decision_pairwise(self, cls_score, thr):
        pred_decision = cls_score - thr
        # pred_decisions = []
        # # pred_scores = torch.sigmoid(cls_score)
        # for data_sample, score in zip(thr, data_samples):
        #     Thr = data_sample.disease_count
        #     # sorted_values, sorted_indices = torch.sort(count)
            
        #     # gumbel_softmax_output = torch.nn.functional.gumbel_softmax(count, tau=1.0, hard=True)
        #     # range_index = torch.arange(len(data_sample.disease_count), device=count.device)
        #     # soft_indices = (gumbel_softmax_output * range_index.float()).sum()
        #     # # _, count_labels = torch.topk(count,k=1,dim=0, largest=True, sorted=False)
        #     # # Thr, _ = score.topk(count.item())
        #     # _, topk_indices = torch.topk(score, k=soft_indices.view(-1)[0], dim=0, largest=True, sorted=False)
        #     # # Thr, _ = score.topk(count.item()) 
        #     # Thr = torch.gather(score, dim=0, index=topk_indices)
        #     # if Thr.numel()==0:
        #     #     topk_values, _ = torch.topk(score, k=1, dim=0, largest=True, sorted=False)
        #     #     Thr = topk_values + 1e-2
        #     # Decision_score = torch.sigmoid(score - Thr)
        #     Decision_score = score - Thr
        #     pred_decisions.append(Decision_score)
        #     # pred_decision.expend(Decision_score)
        # pred_decision = torch.stack(pred_decisions)
        return pred_decision
    # def log_sum_exp_pairwise_loss(self, y_pred, y_true):
    #     """
    #     Compute the Log-Sum-Exp Pairwise (LSEP) loss.

    #     Args:
    #         y_pred (torch.Tensor): Predicted scores (batch_size, n_labels)
    #         y_true (torch.Tensor): Ground truth labels (batch_size, n_labels)

    #     Returns:
    #         torch.Tensor: Computed LSEP loss.
    #     """
    #     batch_size, n_labels = y_pred.shape
    #     # y_pred = torch.sigmoid(y_pred)
    #     # Expand dimensions for pairwise computation
    #     y_pred_i = y_pred.unsqueeze(2)  # Shape: (batch_size, n_labels, 1)
    #     y_pred_j = y_pred.unsqueeze(1)  # Shape: (batch_size, 1, n_labels)
        
    #     y_true_i = y_true.unsqueeze(2)  # Shape: (batch_size, n_labels, 1)
    #     y_true_j = y_true.unsqueeze(1)  # Shape: (batch_size, 1, n_labels)
        
    #     # Compute pairwise differences
    #     pairwise_diff = y_pred_i - y_pred_j  # Shape: (batch_size, n_labels, n_labels)
        
    #     # Compute the pairwise loss mask (only for (i, j) pairs with y_i=1 and y_j=0)
    #     mask = (y_true_i == 1) & (y_true_j == 0)  # Shape: (batch_size, n_labels, n_labels)
        
    #     # Apply weights if provided
    #     if self.Ranking_weight:
    #         weights = self.weight.view(1, -1, 1)  # Reshape weights for broadcasting
    #         mask = mask * weights
        
    #     # Compute LSEP loss
    #     exp_term = torch.exp(pairwise_diff)
    #     masked_exp_term = mask * exp_term  # Apply the mask
    #     sum_exp_term = torch.sum(masked_exp_term, dim=(1, 2))  # Sum over pairs
        
    #     loss = torch.log(1 + sum_exp_term)  # Compute log-sum-exp loss
    #     return loss.mean()  # Return the mean loss over the batch


    def log_sum_exp_pairwise_loss(self, cls_score, labels):
        """
        Log-Sum-Exp Pairwise Loss implementation in PyTorch.
        
        Args:
            predictions (torch.Tensor): Model predictions, shape [batch_size, num_classes].
            labels (torch.Tensor): Ground truth binary labels, shape [batch_size, num_classes].
            size (int): Size for averaging the loss (e.g., batch size).
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        predictions = torch.sigmoid(cls_score)
        # predictions = cls_score
        loss_op = 0.0
        Batch_size = predictions.shape[0]
        for i in range(Batch_size):
            # Extract positive and negative predictions based on labels
            positive = predictions[i][labels[i] == 1.0]  # Positive predictions
            negative = predictions[i][labels[i] == 0.0]  # Negative predictions

            # Compute pairwise differences and exponentiation
            exp_sub = torch.exp(negative[:, None] - positive[None, :])  # Broadcast subtraction
            exp_sum = torch.sum(exp_sub)  # Sum all pairwise exponential differences
            
            # Add log(1 + sum(exp(...))) to the loss
            loss_op += torch.log1p(exp_sum)  # log1p(x) = log(1 + x) for numerical stability

        return loss_op / Batch_size   # Return average loss

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
        cls_score = self(feats)
        if self.Voting:
            Thr = self.Votingforward(feats)
            # kwargs['Thr'] = Thr
            cls_score = cls_score - Thr
       
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(cls_score, data_samples)
        return predictions
    
    def _get_predictions(self, cls_score: torch.Tensor,
                         data_samples: List[DataSample]):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        if self.task == 'multi-label':
            pred_scores = torch.sigmoid(cls_score)
        else:  # 'binary-class'
            pred_scores = torch.softmax(cls_score, dim=1)

        if data_samples is None:
            data_samples = [DataSample() for _ in range(cls_score.size(0))]
        for data_sample, score in zip(data_samples, pred_scores):
            if 'disease_count' in data_sample:
                # Thr = torch.sigmoid(data_sample.disease_count.clone())
                Thr = data_sample.disease_count.clone()
                # count_labels = count.argmax(dim=0, keepdim=True)
                # if count.numel()==0:
                #     label = count
                # else:
                score = torch.sigmoid(score-Thr)
                # label = torch.where(score >= Thr)[0]
                # label = torch.where(score >= self.thr)[0]

                # label = (score > Thr).nonzero(as_tuple=True)[0]
                # _, label = score.topk(count_labels.item())
            if self.thr is not None:
                # a label is predicted positive if larger than thr
                label = torch.where(score >= self.thr)[0]
            else:
                # top-k labels will be predicted positive for any example
                _, label = score.topk(self.topk)
            data_sample.set_pred_score(score).set_pred_label(label)

        return data_samples
