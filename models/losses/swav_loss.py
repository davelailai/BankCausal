# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmengine.dist import all_reduce
from mmengine.model import BaseModule
from mmpretrain.structures import DataSample
import torch.nn.functional as F


from mmpretrain.registry import MODELS


@torch.no_grad()
def distributed_sinkhorn(out: torch.Tensor, sinkhorn_iterations: int,
                         world_size: int, epsilon: float, alpha: float = 0.1, beta: float = 0.3) -> torch.Tensor:
    """Apply the distributed sinkhorn optimization on the scores matrix to find the assignments."""
    eps_num_stab = 1e-12
    Q = torch.exp(out / epsilon).t()  # Q: K x B_local
    B = Q.shape[1] * world_size       # Total samples across all processes
    K = Q.shape[0]                    # Number of prototypes

    # Initial normalization to make global sum = 1
    sum_Q = torch.sum(Q)
    all_reduce(sum_Q)  # Synchronize sum across all processes
    Q /= sum_Q

    for _ in range(sinkhorn_iterations):
        # Row normalization (with alpha relaxation)
        u = torch.sum(Q, dim=1, keepdim=True)
        all_reduce(u)  # 跨进程同步行和
        if torch.any(u < eps_num_stab):
            Q += eps_num_stab
            u = torch.sum(Q, dim=1, keepdim=True)
            all_reduce(u)
        
        Q /= u ** (1 - alpha)  # 软约束归一化
        Q /= K ** alpha  # 目标分布 (1/K)^alpha

        v = torch.sum(Q, dim=0, keepdim=True)  # 1 x B_local
        all_reduce(v)  # 直接求和得到全局列总和（1 x 1）
        # Beta松弛：在均匀分布与当前全局列和之间插值
        Q = Q / (v.clamp(min=eps_num_stab) ** (1 - beta))
        Q /= B ** beta  # 等效于混合目标分布 1/B_total

        sum_Q = torch.sum(Q)
        all_reduce(sum_Q)
        Q *= B / sum_Q 

    return Q.t()


class MultiPrototypes(BaseModule):
    """Multi-prototypes for SwAV head.

    Args:
        output_dim (int): The output dim from SwAV neck.
        num_prototypes (List[int]): The number of prototypes needed.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 output_dim: int,
                 num_prototypes: List[int],
                 init_cfg: Optional[Union[List[dict], dict]] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(num_prototypes, list)
        self.num_heads = len(num_prototypes)
        for i, k in enumerate(num_prototypes):
            self.add_module('prototypes' + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run forward for every prototype."""
        out = []
        for i in range(self.num_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


@MODELS.register_module()
class MultiLabelSwAVLoss(BaseModule):
    """The Loss for SwAV.

    This Loss contains clustering and sinkhorn algorithms to compute Q codes.
    Part of the code is borrowed from `script
    <https://github.com/facebookresearch/swav>`_.
    The queue is built in `engine/hooks/swav_hook.py`.

    Args:
        feat_dim (int): feature dimension of the prototypes.
        sinkhorn_iterations (int): number of iterations in Sinkhorn-Knopp
            algorithm. Defaults to 3.
        epsilon (float): regularization parameter for Sinkhorn-Knopp algorithm.
            Defaults to 0.05.
        temperature (float): temperature parameter in training loss.
            Defaults to 0.1.
        crops_for_assign (List[int]): list of crops id used for computing
            assignments. Defaults to [0, 1].
        num_crops (List[int]): list of number of crops. Defaults to [2].
        num_prototypes (int): number of prototypes. Defaults to 3000.
        init_cfg (dict or List[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 feat_dim: int,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 temperature: float = 0.1,
                 crops_for_assign: List[int] = [0, 1],
                 num_crops: List[int] = [2],
                 num_prototypes: int = 3000,
                 weight: int = 1.0,
                 with_sinkhorn: bool = True,
                 with_avePrior: bool = True,
                 alpha: float = 0.1,
                 beta: float = 0.3,
                 init_cfg: Optional[Union[List[dict], dict]] = None):
        super().__init__(init_cfg=init_cfg)
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.temperature = temperature
        self.crops_for_assign = crops_for_assign
        self.num_crops = num_crops
        self.use_queue = False
        self.queue = None
        self.label_queue = None
        self.weight = weight
        self.data_sample = None
        self.with_sinkhorn = with_sinkhorn
        self.with_avePrior = with_avePrior
        self.alpha = alpha
        self.beta = beta
        # self.prior= torch.ones(num_prototypes) / num_prototypes
        
        # self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32), requires_grad=False)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.layer_norm = torch.nn.LayerNorm(feat_dim)
        # prototype layer
        # prototypes = torch.randn(num_prototypes, feat_dim, dtype=torch.float32)
        # prototypes = F.normalize(prototypes, dim=1)  # L2 normalize
        # self.prototypes = nn.Parameter(prototypes, requires_grad=True)
        self.prototypes = self.orthogonal_l2_normalized_init(num_prototypes, feat_dim)
        # self.prototypes = nn.Parameter(torch.randn(num_prototypes, feat_dim, dtype=torch.float32), requires_grad=True)
        if self.with_avePrior:
            self.register_buffer("prior", torch.ones(num_prototypes) / num_prototypes)
        else:
            self.prior = nn.Parameter(torch.ones(num_prototypes) / num_prototypes, requires_grad=True)
        # self.prototypes = nn.Parameter(torch.zeros(num_prototypes, feat_dim, dtype=torch.float32), requires_grad=True)
        
        # self.prototypes = None
        # if isinstance(num_prototypes, list):
        #     self.prototypes = MultiPrototypes(feat_dim, num_prototypes)
        # elif num_prototypes > 0:
        #     self.prototypes = nn.Linear(feat_dim, num_prototypes, bias=False)
        # assert self.prototypes is not None
    def orthogonal_l2_normalized_init(self, num_prototypes, feat_dim, device='cuda'):
        # 初始化：用标准正态生成 feat_dim 个正交向量
        proto = torch.randn(feat_dim, num_prototypes, device=device)  # 注意 shape 是 [D, K]

        # 做 QR 分解（得到 [D, K] 的正交列向量）
        q, _ = torch.linalg.qr(proto)

        # 转置回来，得到 [K, D]
        proto = q.T  # [num_prototypes, feat_dim]

        # L2 normalize 每一行
        proto = F.normalize(proto, p=2, dim=1)

        return nn.Parameter(proto, requires_grad=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function of SwAV loss.

        Args:
            x (torch.Tensor): NxC input features.
        Returns:
            torch.Tensor: The returned loss.
        """
        # normalize the prototypes
        # with torch.no_grad():
        #     w = nn.functional.normalize(self.prototypes, p=2, dim=1)  # L2 normalization
        #     self.prototypes.copy_(w)  # Keep it as a trainable parameter
        # with torch.no_grad():
        #     self.prototypes.data = nn.functional.normalize(self.prototypes.data, p=2, dim=1)
         # normalize the prototypes
        # with torch.no_grad():
        #     w = self.prototypes.clone()
        #     w = nn.functional.normalize(w, dim=1, p=2)
        #     self.prototypes.copy_(w)
        # normalized_prototypes = nn.functional.normalize(self.prototypes, p=2, dim=1)
        if self.data_sample is not None:
            labels = torch.stack([data_sample.gt_score for data_sample in self.data_sample])
            # intersection = labels @ labels.t().float()
            # norm = torch.sqrt(labels.sum(dim=1, keepdim=True))  # 每个样本的标签数 [N,1]
            # # 防止零除（处理全零标签）
            # valid_mask = (norm > 0).float()
            # norm = norm * valid_mask + (1 - valid_mask)  # 全零样本的norm设为1
            # sim = intersection / (norm @ norm.T)
            # label_sim = sim.clamp(min=0, max=1)
            # x = x + 0.5 * (label_sim @ x)
            # x = self.layer_norm(x)
   
        x = nn.functional.normalize(x, dim=1, p=2)
        output = torch.mm(x, self.prototypes.t())  # 当前特征的相似度计算（含梯度）

        # 分离当前特征用于队列更新
        embedding = x.detach()
        bs = int(embedding.size(0) / sum(self.num_crops))
        loss=0
        for i, crop_id in enumerate(self.crops_for_assign):
            with torch.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)].detach()
                if self.queue is not None:
                    if self.use_queue or not torch.all(self.queue[i,-1, :] == 0):
                        self.use_queue = True
                        queue_output = torch.mm(self.queue[i], self.prototypes.t())  # 队列特征相似度（含梯度）
                        out = torch.cat((queue_output, out))
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) *bs]
                    if self.data_sample is not None:
                        self.label_queue[bs:] = self.label_queue[:-bs].clone()
                        self.label_queue[:bs] = labels.detach()
                    
        
        # # 更新队列（先进先出，确保新特征已分离）
        #     if self.queue is not None:
        #         self.queue = torch.roll(self.queue, shifts=-x.size(0), dims=0)
        #         self.queue[-x.size(0):] = embedding
        #         if self.data_sample is not None:
        #             self.label_queue=torch.roll(self.label_queue, shifts=-x.size(0), dims=0)
        #             self.label_queue[-x.size(0):] = labels.detach()

                # Sinkhorn-Knopp分配
                q = distributed_sinkhorn(out, 
                                        self.sinkhorn_iterations,
                                        self.world_size,
                                        self.epsilon,
                                        alpha=self.alpha,
                                        beta=self.beta)
            subloss=0
            for v in np.delete(np.arange(np.sum(self.num_crops)), crop_id):
                x = output[bs * v:bs * (v + 1)] / self.temperature
                subloss -= torch.mean(
                    torch.sum(q[-bs:] * nn.functional.log_softmax(x, dim=1), dim=1))
                loss += subloss / (np.sum(self.num_crops) - 1)

        # 计算对比损失
        # assert not torch.isnan(q).any()
        # log_prob = nn.functional.log_softmax(output / self.temperature, dim=1)
        # assert not torch.isnan(log_prob).any()
        # if self.prior is not None:
        if self.with_avePrior:
            if not torch.all(self.queue[i,-1, :] == 0):
            # if self.prototypes.requires_grad and not torch.all(self.queue[i,-1, :] == 0):
                self.update_prior(q,bs), # q[-bs:]
            # self.prior = self.update_prior(q).to(x.device)
    
        loss /= len(self.crops_for_assign)
        # loss = -torch.mean(torch.sum(q * log_prob, dim=1))
        # print(self.weight)

        return self.weight * loss
    
    # def update_prior(self, batch_q, momentum=0.8, temperature=0.5):
    #     batch_prior = torch.mean(batch_q, dim=0)
    #     # sharpened_prior = batch_prior.pow(1/temperature)  # 温度越低，分布越尖锐
    #     # sharpened_prior /= sharpened_prior.sum()
    #     sharpened_prior = batch_prior
    #     updated_prior = momentum * self.prior + (1 - momentum) * sharpened_prior
    #     self.prior.copy_(updated_prior / updated_prior.sum())
    # def update_prior(self, batch_q, momentum=0.2):
    #     batch_prior = torch.mean(batch_q, dim=0)
    #     updated_prior = momentum * self.prior + (1 - momentum) * batch_prior
    #     self.prior.copy_(updated_prior / updated_prior.sum())
    # def update_prior(self, batch_q,momentum = 0.5):
    #     batch_prior = torch.mean(batch_q, dim=0)  # [K]
    #     updated_prior = momentum * self.prior.to(batch_prior.device) + (1 - momentum) * batch_prior
    #     updated_prior = updated_prior / updated_prior.sum()  # Normalize
    #     return updated_prior
    def update_prior(self, q: torch.Tensor, batch_size: int, momentum: float = 0.8, weight_batch: float = 0.6):
        queue_size = q.shape[0] - batch_size
        assert queue_size >= 0, "queue size must be >= 0"

        alpha_batch = q[-batch_size:].mean(dim=0)
        if queue_size > 0:
            alpha_queue = q[:queue_size].mean(dim=0)
            alpha_star = weight_batch * alpha_batch + (1 - weight_batch) * alpha_queue
        else:
            alpha_star = alpha_batch

        if hasattr(self, "prior") and self.prior is not None:
            updated_prior = momentum * self.prior + (1 - momentum) * alpha_star
        else:
            updated_prior = alpha_star

        self.prior.copy_(updated_prior / updated_prior.sum())

    
    
