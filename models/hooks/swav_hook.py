# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.device import get_device
from mmengine.dist import get_rank, get_world_size, is_distributed
from mmengine.hooks import Hook
from mmengine.logging import MMLogger

from mmpretrain.registry import HOOKS
from mmpretrain.utils import get_ori_model
from sklearn.cluster import KMeans
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# import umap
def compute_colors_from_multilabels(Y):
        """
        输入：Y 是 (n_samples, n_labels) 的多标签 0/1 矩阵
        输出：每个样本对应的 RGB 颜色，归一化在 [0, 1]
        """
        from sklearn.preprocessing import MinMaxScaler
        pca = PCA(n_components=3)
        Y_pca = pca.fit_transform(Y)

        # 归一化到 0-1 范围作为 RGB
        Y_rgb = MinMaxScaler().fit_transform(Y_pca)
        return Y_rgb  # shape: (n_samples, 3)

def cluster_plot(features, labels, prototypes, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    # unique_labels = list(set(labels))
    # palette = sns.color_palette("hsv", len(unique_labels))
    colors = compute_colors_from_multilabels(labels)
    # color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot features
    ax.scatter(features[:, 0], features[:, 1], c=colors, s=10)
    # for label in unique_labels:
    #     idx = [i for i, l in enumerate(labels) if l == label]
    #     points = features[idx]
    #     ax.scatter(points[:, 0], points[:, 1], label=label, c=[color_map[label]] * len(idx), s=10)

    # Plot prototypes
    if prototypes is not None:
        ax.scatter(prototypes[:, 0], prototypes[:, 1], c='black', marker='X', s=100, label='Prototypes')

        # Draw lines from each point to its closest prototype
        for feat in features:
            # Compute distance to all prototypes
            distances = np.linalg.norm(prototypes - feat, axis=1)
            closest_proto_idx = np.argmin(distances)
            closest_proto = prototypes[closest_proto_idx]
            ax.plot([feat[0], closest_proto[0]], [feat[1], closest_proto[1]], 'k-', linewidth=0.3, alpha=0.3)

    ax.legend(loc='best', fontsize='small')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
def force_directed_graph_plot(features_np, labels_np, reduced_proto, save_path, top_k=2, use_softmax=True):
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import softmax

    # 确保 numpy 格式
    features_np = np.array(features_np)
    labels_np = np.array(labels_np)

    # # 如果是字符串标签（多标签编码后的），先转成数字 ID
    # if labels_np.dtype.type is np.str_:
    #     unique_labels = sorted(set(labels_np))
    #     label_to_id = {label: i for i, label in enumerate(unique_labels)}
    #     numeric_labels = np.array([label_to_id[lbl] for lbl in labels_np])
    # else:
    #     # 如果是 one-hot 多标签，转 argmax；如果是单标签就直接用
    #     if labels_np.ndim > 1:
    #         node_rgb_colors = self.compute_colors_from_multilabels(labels_np)
    #         numeric_labels = np.argmax(labels_np, axis=1)
    #     else:
    #         numeric_labels = labels_np

    # 如果有 TTA 导致 feature 多于标签，尝试聚合
    if features_np.shape[0] > labels_np.shape[0]:
        tta_times = features_np.shape[0] // labels_np.shape[0]
        assert features_np.shape[0] % labels_np.shape[0] == 0, "features 数量无法整除 labels"
        features_np = features_np.reshape(labels_np.shape[0], tta_times, -1).mean(axis=1)

    # 限制样本数量
    sample_limit = 2000
    if features_np.shape[0] > sample_limit:
        sample_indices = np.random.choice(features_np.shape[0], sample_limit, replace=False)
        features_np = features_np[sample_indices]
        labels_np = labels_np[sample_indices]
    
    if use_softmax:
        features_np = softmax(features_np, axis=1)
    else:
        features_np = np.maximum(0, features_np)  # 裁剪负值

    # 计算 RGB 颜色（基于多标签 PCA）
    node_rgb_colors = compute_colors_from_multilabels(labels_np)
    node_rgb_colors = np.clip(node_rgb_colors, 0, 1)
    num_centers = features_np.shape[1]
    G = nx.Graph()

    # 添加中心点
    for c in range(num_centers):
        G.add_node(f"C{c}", type='center', weight=0)

    # 添加样本点和边
    for i in range(features_np.shape[0]):
        sample_node = f"S{i}"
        G.add_node(sample_node, type='sample')
        row = features_np[i]
        top_indices = row.argsort()[-top_k:]
        for c in top_indices:
            weight = row[c]
            G.add_edge(sample_node, f"C{c}", weight=weight)
            # 更新该中心点的权重（连接的权重之和）
            G.nodes[f"C{c}"]['weight'] += weight

    # 可视化
    # pos = nx.spring_layout(G)  # 使用 kamada_kawai_layout，可能会更好
    pos = nx.spring_layout(G, seed=42, center=(0, 0), scale=1.0)
    # 准备颜色与大小
    sample_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "sample"]
    center_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "center"]
    node_colors = node_rgb_colors[:len(sample_nodes)]
    # cmap = plt.cm.get_cmap("tab20", np.max(numeric_labels) + 1)
    center_sizes = [20 * G.nodes[n]['weight'] for n in center_nodes]

    # # 绘图
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_margin = (max(x_vals) - min(x_vals)) * 0.1
    y_margin = (max(y_vals) - min(y_vals)) * 0.1
    plt.figure(figsize=(14, 12))
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=sample_nodes,
        node_color=node_colors,
        node_size=30,
        alpha=0.8
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=center_nodes,
        node_color="red",
        node_size=center_sizes,
        alpha=0.8
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    plt.xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
    plt.ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)
    plt.title("Force-Directed Graph: Soft Assignment (Samples → Confounders)", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@HOOKS.register_module()
class MultiLabelSwAVHook(Hook):
    """Hook for SwAV.

    This hook builds the queue in SwAV according to ``epoch_queue_starts``.
    The queue will be saved in ``runner.work_dir`` or loaded at start epoch
    if the path folder has queues saved before.

    Args:
        batch_size (int): the batch size per GPU for computing.
        epoch_queue_starts (int, optional): from this epoch, starts to use the
            queue. Defaults to 15.
        crops_for_assign (list[int], optional): list of crops id used for
            computing assignments. Defaults to [0, 1].
        feat_dim (int, optional): feature dimension of output vector.
            Defaults to 128.
        queue_length (int, optional): length of the queue (0 for no queue).
            Defaults to 0.
        interval (int, optional): the interval to save the queue.
            Defaults to 1.
        frozen_layers_cfg (dict, optional): Dict to config frozen layers.
            The key-value pair is layer name and its frozen iters. If frozen,
            the layers don't need gradient. Defaults to dict().
    """

    def __init__(
        self,
        batch_size: int,
        epoch_queue_starts: Optional[int] = 15,
        crops_for_assign: Optional[List[int]] = [0, 1],
        feat_dim: Optional[int] = 128,
        queue_length: Optional[int] = 0,
        interval: Optional[int] = 1,
        frozen_epoch_cfg: Optional[Dict] = dict(),
        loss_decay: Optional[Dict] = dict()
    ) -> None:
        self.batch_size = batch_size * get_world_size()
        self.epoch_queue_starts = epoch_queue_starts
        self.crops_for_assign = crops_for_assign
        self.feat_dim = feat_dim
        self.queue_length = queue_length
        self.interval = interval
        self.frozen_epoch_cfg = frozen_epoch_cfg
        self.loss_decay =loss_decay
        self.requires_grad = True
        self.queue = None

    def before_run(self, runner) -> None:
        """Check whether the queues exist locally or not."""
        if is_distributed():
            self.queue_path = osp.join(runner.work_dir,
                                       'queue' + str(get_rank()) + '.pth')
            self.label_queue_path = osp.join(runner.work_dir,
                                       'label_queue' + str(get_rank()) + '.pth')
        else:
            self.queue_path = osp.join(runner.work_dir, 'queue.pth')
            self.label_queue_path = osp.join(runner.work_dir, 'label_queue.pth')
            
            
        # load the queues if queues exist locally
        if osp.isfile(self.queue_path):
            self.queue = torch.load(self.queue_path)['queue']
            self.label_queue = torch.load(self.label_queue_path)['label_queue']
            # print(runner.model)
            if hasattr(runner.model, 'module'):
                if hasattr(runner.model.module, 'module'):  # Check if wrapped by TTA
                    model = runner.model.module.module
                else:
                    model = runner.model.module
            else:  # Fallback to original model
                model = runner.model
            # print(model)
           
            get_ori_model(model).head.swav_loss_module.queue = self.queue
            get_ori_model(model).head.swav_loss_module.label_queue = self.label_queue
            MMLogger.get_current_instance().info(
                f'Load queue from file: {self.queue_path}')
        # the queue needs to be divisible by the batch size
        self.queue_length -= self.queue_length % self.batch_size

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        """Freeze layers before specific iters according to the config."""
        prototypes = get_ori_model(runner.model).head.swav_loss_module.prototypes
        for layer, frozen_epoch in self.frozen_epoch_cfg.items():
            if runner.epoch >= frozen_epoch:
                for name, p in get_ori_model(runner.model).named_parameters():
                    if layer in name:
                        p.requires_grad = False
            else:
                for name, p in get_ori_model(runner.model).named_parameters():
                        if layer in name:
                            p.requires_grad = True
            # if self.loss_decay:
            #     if runner.epoch <= self.loss_decay.total_epochs:
            #         for name, p in get_ori_model(runner.model).named_parameters():
            #             if layer in name:
            #                 p.requires_grad = True
            #     else:
            #         for name, p in get_ori_model(runner.model).named_parameters():
            #             if layer in name:
            #                 p.requires_grad = False
               
        # Get the current prototypes tensor
        # EMA更新（只在未冻结时）
        model = get_ori_model(runner.model)
        prototypes = model.head.swav_loss_module.prototypes
        if any(runner.epoch < frozen_epoch for frozen_epoch in self.frozen_epoch_cfg.values()):
            with torch.no_grad():
                current_proto = prototypes.data.clone()
                current_proto_norm = torch.nn.functional.normalize(current_proto, p=2, dim=1)
                prototypes.data.copy_(current_proto_norm)
        
        # # On the first iteration, just save a copy
        if runner.iter%100==0:
            if not hasattr(self, 'prev_prototypes'):
                self.prev_prototypes = prototypes.detach().clone()
                print("Initialized previous prototypes.")
                return
            
            # Compare current prototypes with previous ones
            changed = not torch.equal(prototypes.detach(), self.prev_prototypes)
            
            if changed:
                print("Prototypes have changed since last iteration.")
            else:
                print("Prototypes have NOT changed since last iteration.")
            
            # Update stored prototypes for next iteration
            self.prev_prototypes = prototypes.detach().clone()
        # prototypes = get_ori_model(runner.model).head.swav_loss_module.prototypes
        #
            # prototypes.data.copy_(normalized)
           
        get_ori_model(runner.model).head.swav_loss_module.data_sample=data_batch['data_samples']


    def before_train_epoch(self, runner) -> None:
        """Check the queues' state."""
        # optionally starts a queue
        if self.queue_length > 0 \
            and runner.epoch >= self.epoch_queue_starts \
                and self.queue is None:

            self.queue = torch.zeros(
                len(self.crops_for_assign),
                self.queue_length // runner.world_size,
                self.feat_dim,
                device=get_device(),
            )
            self.label_queue = torch.zeros(
                # len(self.crops_for_assign),
                self.queue_length // runner.world_size,
                get_ori_model(runner.model).head.num_class,
                device=get_device(),
            )
            get_ori_model(runner.model).head.swav_loss_module.queue = self.queue
            get_ori_model(runner.model).head.swav_loss_module.label_queue = self.label_queue
            get_ori_model(runner.model).head.swav_loss_module.use_queue = False
        # set the boolean type of use_the_queue
        get_ori_model(runner.model).head.current_epoch = runner.epoch
        get_ori_model(runner.model).head.warmup_epoch = get_ori_model(runner.model).warmup_epoch
       
    def after_train_epoch(self, runner) -> None:
        """Save the queues locally."""
        self.queue = get_ori_model(runner.model).head.swav_loss_module.queue
        self.label_queue = get_ori_model(runner.model).head.swav_loss_module.label_queue
        get_ori_model(runner.model).head.swav_loss_module.data_sample = None
        # if self.queue is not None and runner.epoch == self.frozen_epoch_cfg.prototypes:
        #     queue_np = self.queue.cpu().numpy()  # Convert to NumPy for k-means
        #     # label_np = self.label_queue.cpu().numpy
        #     # Apply k-means clustering
        #     num_clusters = get_ori_model(runner.model).head.swav_loss_module.prototypes.shape[0]  # Ensure valid cluster count
        #     kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        #     cluster_assignments=kmeans.fit_predict(queue_np)
        #     cluster_assignments = torch.tensor(cluster_assignments, dtype=torch.long, device=self.queue.device)
        #     # cluster_assignments=kmeans.fit_predict(queue_np)
        #     # weights = queue_labels @ queue_labels.T.mean(dim=1)  # 标签相似度作为权重
        #     # kmeans = WeightedKMeans(n_clusters=K)
        #     # kmeans.fit(queue_features, sample_weight=weights)
        #     # new_prototypes = kmeans.cluster_centers_

        #     new_weight = torch.from_numpy(kmeans.cluster_centers_).to(dtype=torch.float32, device=self.queue.device)
        #     # eps = 1e-6  # Small value to prevent division by zero
        #     # new_weight = new_weight / (torch.linalg.norm(new_weight, axis=1, keepdims=True) + eps)
        #     # eps = 1e-6  # Prevent division by zero
        #     # new_weight = new_weight / (torch.norm(new_weight, p=2, dim=1, keepdim=True) + eps)
        #     # new_weight = new_weight - new_weight.mean(dim=0)
        #     # new_weight = new_weight / (new_weight.std(dim=0) + 1e-6)
        #     new_weight = F.normalize(new_weight, p=2, dim=1)

        #     assert not torch.isnan(new_weight).any()

        #     get_ori_model(runner.model).head.swav_loss_module.prototypes.data.copy_(new_weight)

            # class_counts = torch.zeros((num_clusters), device=self.queue.device)
            # for k in range(num_clusters):
            #     mask = (cluster_assignments == k)
            #     class_counts[k] = mask.sum()
            # # runner.logger.info(
            # # f"Epoch {runner.epoch} Prototype Prior - "f"{queue_np[1:7,:]}")
            # prior = torch.softmax(class_counts, dim=0)  #
            # get_ori_model(runner.model).head.swav_loss_module.prior=prior

            # runner.logger.info("Updated loss weight using k-means clustering.")
        
        if self.loss_decay and runner.epoch >= self.loss_decay.start_epoch:
            start_w = self.loss_decay.start_weight
            end_w = self.loss_decay.end_weight
            decay_begin = self.loss_decay.start_epoch
            total_epochs = self.loss_decay.total_epochs
            decay_duration = total_epochs - decay_begin
            decay_step = runner.epoch - decay_begin
            decay_progress = min(decay_step / decay_duration, 1.0)
            if self.loss_decay.decay_type == 'linear':
                new_weight = start_w - (start_w - end_w) * decay_progress
            elif self.loss_decay.decay_type == 'exponential':
                decay_rate = (end_w / start_w) ** (1 / decay_duration)
                new_weight = start_w * (decay_rate ** decay_step)
            new_weight = max(self.loss_decay.end_weight, new_weight)
            # print("Before update:", get_ori_model(runner.model).head.swav_loss_module.weight)  # Debugging
            # with torch.no_grad():
            #     get_ori_model(runner.model).head.swav_loss_module.weight = new_weight
            #     runner.model.head.swav_loss_module.weight.copy_(torch.tensor(new_weight, dtype=torch.float32, device=runner.model.head.swav_loss_module.weight.device))
            get_ori_model(runner.model).head.swav_loss_module.weight = new_weight
            runner.logger.info(f"Updated loss weight to: {get_ori_model(runner.model).head.swav_loss_module.weight}") 
            # print("After update:", get_ori_model(runner.model).head.swav_loss_module.weight)  # Debugging
            # 1. 获取原型权重矩阵
        with torch.no_grad():
            prototypes = get_ori_model(runner.model).head.swav_loss_module.prototypes.clone()
        
        # if self.queue is not None and not torch.all(self.queue[0,-1, :] == 0) and runner.epoch%9==0:
        #     proto_np = F.normalize(prototypes, dim=1)
        #     similarity = torch.mm(self.queue[0], proto_np.t()) 
            
        #     # if self.label_queue.dim() > 1:
        #     #     labels_np = self.encode_multi_label(self.label_queue)
        #     #     # labels_np = labels.argmax(dim=1).numpy()
        #     # else:
        #     labels_np = self.label_queue.cpu().numpy()

        #     reduced_proto = None
        #     features_np = self.queue[0].cpu().detach().numpy()
        #     reduced_feats = self._reduce(features_np)
        #     save_path = osp.join(runner.work_dir,'TSNE')
        #     os.makedirs(save_path, exist_ok=True)
        #     cluster_plot(reduced_feats, labels_np, reduced_proto, os.path.join(save_path,'Clustering_vis_epoch'+str(runner.epoch)+'.png'))
        #     runner.logger.info(f'[PrototypeClusteringHook] Visualization saved to {save_path}')
        #     similarity_np = similarity.cpu().detach().numpy()
            
        #     save_path = osp.join(runner.work_dir,'Force-Directed Graph')
        #     os.makedirs(save_path, exist_ok=True)
        #     force_directed_graph_plot(similarity_np,labels_np,reduced_proto,os.path.join(save_path,'Clustering_vis_epoch'+str(runner.epoch)+'.png'))
        #     # self._plot(reduced_feats, labels_np, reduced_proto, self.save_path)
        #     runner.logger.info(f'[Force-Directed Graph] Visualization saved to {save_path}')
        # 2. 计算余弦相似度矩阵
        with torch.no_grad():
            # 归一化原型向量
            normalized_proto = F.normalize(prototypes, p=2, dim=1)
            
            # 计算全连接相似度矩阵
            similarity_matrix = torch.mm(normalized_proto, normalized_proto.t())
            
            # 排除对角线元素（自身相似度）
            mask = torch.eye(similarity_matrix.size(0), 
                            dtype=torch.bool, 
                            device=similarity_matrix.device)
            filtered_sim = similarity_matrix[~mask].view(
                            similarity_matrix.size(0), -1)
        
        # 3. 计算统计指标
        stats = {
            # 平均相似度（排除对角线）
            'proto/mean_sim': filtered_sim.mean().item(),
            
            # 最大相似度（排除对角线）
            'proto/max_sim': filtered_sim.max(dim=1)[0].mean().item(),
            
            # 最小相似度（排除对角线）
            'proto/min_sim': filtered_sim.min(dim=1)[0].mean().item(),
            
            # # 相似度矩阵直方图
            # 'proto/sim_hist': wandb.Histogram(similarity_matrix.cpu().numpy())
        }
        
        # 4. 记录到日志系统
        # runner.logger.info(
        #     f"Epoch {runner.epoch} Prototype Prior - "f"{self.queue[1:7,:]}")
        runner.logger.info(
            f"Epoch {runner.epoch} Prototype Prior - "f"{get_ori_model(runner.model).head.swav_loss_module.prior.squeeze().cpu().detach().numpy()}")
        runner.logger.info(
            f"Epoch {runner.epoch} Prototype Similarity - "
            f"Mean: {stats['proto/mean_sim']:.4f}, "
            f"Max: {stats['proto/max_sim']:.4f}, "
            f"Min: {stats['proto/min_sim']:.4f}"
        ) 
        if self.queue is not None and self.every_n_epochs(
                runner, self.interval):
            torch.save({'queue': self.queue}, self.queue_path)
            torch.save({'label_queue': self.label_queue}, self.label_queue_path)
            queue_mean = self.queue.mean().item()
            queue_nonzero = (self.queue != 0).float().mean().item()
            runner.logger.info(f"Queue Stats - Mean: {queue_mean:.4f}, Non-zero: {queue_nonzero:.4f}")

    # def encode_multi_label(self, labels_tensor):
    #     labels_np = labels_tensor.cpu().numpy()
    #     multi_labels = []

    #     for row in labels_np:
    #         active_indices = [str(i) for i, val in enumerate(row) if val == 1]
    #         if active_indices:
    #             multi_labels.append("_".join(active_indices))
    #         else:
    #             multi_labels.append("none")  # Or a default/fallback class

    #     return multi_labels
    
    def _reduce(self, features, method='tsne',n_components=2,perplexity=30,random_state=42):
        if method == 'tsne':
            reducer = TSNE(n_components=n_components,
                           perplexity=perplexity,
                           random_state=random_state,
                           metric='cosine'
)
        elif method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'umap':
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=15,        # 可调参数
                metric='cosine',       # 适合已归一化特征
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
        return reducer.fit_transform(features)

    def _plot(self, features, labels, prototypes, save_path=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        unique_labels = list(set(labels))
        palette = sns.color_palette("hsv", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot features
        for label in unique_labels:
            idx = [i for i, l in enumerate(labels) if l == label]
            points = features[idx]
            ax.scatter(points[:, 0], points[:, 1], label=label, c=[color_map[label]] * len(idx), s=10)

        # Plot prototypes
        if prototypes is not None:
            ax.scatter(prototypes[:, 0], prototypes[:, 1], c='black', marker='X', s=100, label='Prototypes')

            # Draw lines from each point to its closest prototype
            for feat in features:
                # Compute distance to all prototypes
                distances = np.linalg.norm(prototypes - feat, axis=1)
                closest_proto_idx = np.argmin(distances)
                closest_proto = prototypes[closest_proto_idx]
                ax.plot([feat[0], closest_proto[0]], [feat[1], closest_proto[1]], 'k-', linewidth=0.3, alpha=0.3)

        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()
    def force_directed_graph_plot(self, features_np, labels_np, reduced_proto, save_path, top_k=2, use_softmax=True):
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.special import softmax

        # 确保 numpy 格式
        features_np = np.array(features_np)
        labels_np = np.array(labels_np)

        # 如果是字符串标签（多标签编码后的），先转成数字 ID
        if labels_np.dtype.type is np.str_:
            unique_labels = sorted(set(labels_np))
            label_to_id = {label: i for i, label in enumerate(unique_labels)}
            numeric_labels = np.array([label_to_id[lbl] for lbl in labels_np])
        else:
            # 如果是 one-hot 多标签，转 argmax；如果是单标签就直接用
            if labels_np.ndim > 1:
                numeric_labels = np.argmax(labels_np, axis=1)
            else:
                numeric_labels = labels_np

        # 如果有 TTA 导致 feature 多于标签，尝试聚合
        if features_np.shape[0] > numeric_labels.shape[0]:
            tta_times = features_np.shape[0] // numeric_labels.shape[0]
            assert features_np.shape[0] % numeric_labels.shape[0] == 0, "features 数量无法整除 labels"
            features_np = features_np.reshape(numeric_labels.shape[0], tta_times, -1).mean(axis=1)

        # 限制样本数量
        sample_limit = 1000
        if features_np.shape[0] > sample_limit:
            sample_indices = np.random.choice(features_np.shape[0], sample_limit, replace=False)
            features_np = features_np[sample_indices]
            numeric_labels = numeric_labels[sample_indices]
        
        if use_softmax:
            features_np = softmax(features_np, axis=1)
        else:
            features_np = np.maximum(0, features_np)  # 裁剪负值

        num_centers = features_np.shape[1]

        G = nx.Graph()

        # 添加中心点
        for c in range(num_centers):
            G.add_node(f"C{c}", type='center', weight=0)

        # 添加样本点和边
        for i in range(features_np.shape[0]):
            sample_node = f"S{i}"
            G.add_node(sample_node, type='sample', label=int(numeric_labels[i]))

            row = features_np[i]
            top_indices = row.argsort()[-top_k:]
            for c in top_indices:
                weight = row[c]
                G.add_edge(sample_node, f"C{c}", weight=weight)
                # 更新该中心点的权重（连接的权重之和）
                G.nodes[f"C{c}"]['weight'] += weight

        # 可视化
        pos = nx.kamada_kawai_layout(G)  # 使用 kamada_kawai_layout，可能会更好
        cmap = plt.cm.get_cmap("tab20", np.max(numeric_labels) + 1)

        node_colors = []
        center_sizes = []
        # 计算中心点的大小，根据连接的样本数量和权重调整
        for node, attr in G.nodes(data=True):
            if attr["type"] == "center":
                # 中心点的大小基于该中心点的权重
                center_size = 5 * attr['weight']  # 可以根据需要调整比例
                center_sizes.append(center_size)
            else:
                # 样本节点大小保持不变
                node_colors.append(cmap(attr.get("label", 0)))

        # 绘图
        plt.figure(figsize=(14, 12))
        # 绘制样本节点（小尺寸）
        nx.draw_networkx_nodes(G, pos, nodelist=[node for node, attr in G.nodes(data=True) if attr["type"] == "sample"],
                            node_color=node_colors, node_size=30, alpha=0.8)
        # 绘制聚类中心节点（大尺寸）
        nx.draw_networkx_nodes(G, pos, nodelist=[node for node, attr in G.nodes(data=True) if attr["type"] == "center"],
                            node_color="red", node_size=center_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
        plt.title("Force-Directed Graph: Soft Assignment (Samples → Confounders)", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()





@HOOKS.register_module()
class PrototypeClusteringHook(Hook):
    def __init__(self,
                 save_path='work_dirs/clustering.png',
                 layer_name='backbone',
                 method='tsne',  # 'tsne' or 'pca'
                 with_prototypes=True,
                 proto_attr='head.swav_loss_module.prototypes',
                 perplexity=30,
                 num_crop=10,
                 n_components=2,
                 random_state=42):
        self.save_path = save_path
        self.layer_name = layer_name
        self.method = method
        self.perplexity = perplexity
        self.n_components = n_components
        self.random_state = random_state
        self.with_prototypes = with_prototypes
        self.proto_attr = proto_attr
        self.num_views = num_crop
        self.features = []
        self.labels = []
        self.handle = None
        self.prototypes = None
    def unwrap_model(self, model):
        # 递归解包 TTA, DDP 等 wrapper
        while hasattr(model, 'module') or hasattr(model, 'model'):
            if hasattr(model, 'module'):
                model = model.module
            elif hasattr(model, 'model'):
                model = model.model
        return model

    def before_test(self, runner):
        self.features = []
        self.labels = []
        self.curr_view_idx = 0

        # self.num_views = self.num_crop
        # self.model = runner.model
        self.model = self.unwrap_model(runner.model)
        self._hook_called = False
        target_layer = self._get_layer(self.model, self.layer_name)
        self.handle = target_layer.register_forward_hook(self._hook_fn)

        if self.with_prototypes:
            self.prototypes = self._get_layer(self.model, self.proto_attr)
            if isinstance(self.prototypes, torch.nn.Parameter):
                self.prototypes = self.prototypes.detach().cpu()
            else:
                self.prototypes = self.prototypes.weight.detach().cpu()

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        ds = data_batch['data_samples']
        if isinstance(ds, list) and len(ds) > 0 and isinstance(ds[0], list):
            # TTA: 取第一个视图的标签
            batch_labels = [d.gt_score for d in ds[0]]
        else:
            batch_labels = [d.gt_score for d in ds]
        # if isinstance(data_batch['data_samples'], list):
        #     batch_labels = [ds.gt_score for ds in data_batch['data_samples'][0]]
        # else:
        #     batch_labels = [ds.gt_score for ds in data_batch['data_samples']]
        batch_labels = torch.stack(batch_labels).cpu()
        self.labels.append(batch_labels)

    def after_test(self, runner):
        if self.handle:
            self.handle.remove()

        features = torch.cat(self.features, dim=0)
        labels = torch.cat(self.labels, dim=0)
        features = F.normalize(features, dim=1)
        if self.prototypes is not None:
            proto_np = F.normalize(self.prototypes, dim=1)
            features = torch.mm(features, proto_np.t())  

        # intersection = labels @ labels.t().float()
        # norm = torch.sqrt(labels.sum(dim=1, keepdim=True))  # 每个样本的标签数 [N,1]
        # # 防止零除（处理全零标签）
        # valid_mask = (norm > 0).float()
        # norm = norm * valid_mask + (1 - valid_mask)  # 全零样本的norm设为1
        # sim = intersection / (norm @ norm.T)
        # label_sim = sim.clamp(min=0, max=1)
        # features = features + 0.5 * (label_sim @ features)

        

        # if labels.dim() > 1:
        #     labels_np = self.encode_multi_label(labels)
        #     node_rgb_colors = self.compute_colors_from_multilabels(labels_np)
        #     # labels_np = labels.argmax(dim=1).numpy()
        # else:
        labels_np = labels.numpy()

        features_np = features.numpy()
        reduced_feats = self._reduce(features_np)

        # if self.prototypes is not None:
        #     proto_np = F.normalize(self.prototypes, dim=1).numpy()
        #     # output = torch.mm(x, normalized_prototypes.t())  
        #     reduced_proto = self._reduce(proto_np)
        # else:
        #     reduced_proto = None
        reduced_proto = None
        save_path = osp.join(runner.work_dir,'Force_Directed_Graph')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.force_directed_graph_plot(features_np,labels_np,reduced_proto,os.path.join(save_path, 'Clustering_vis.png'))
        # self._plot(reduced_feats, labels_np, reduced_proto, self.save_path)
        runner.logger.info(f'[PrototypeClusteringHook] Visualization saved to {save_path}')

    def encode_multi_label(self, labels_tensor):
        labels_np = labels_tensor.numpy()
        multi_labels = []

        for row in labels_np:
            active_indices = [str(i) for i, val in enumerate(row) if val == 1]
            if active_indices:
                multi_labels.append("_".join(active_indices))
            else:
                multi_labels.append("none")  # Or a default/fallback class

        return multi_labels
    # def compute_colors_from_multilabels(self,Y):
    #     """
    #     输入：Y 是 (n_samples, n_labels) 的多标签 0/1 矩阵
    #     输出：每个样本对应的 RGB 颜色，归一化在 [0, 1]
    #     """
    #     from sklearn.preprocessing import MinMaxScaler
    #     pca = PCA(n_components=3)
    #     Y_pca = pca.fit_transform(Y)

    #     # 归一化到 0-1 范围作为 RGB
    #     Y_rgb = MinMaxScaler().fit_transform(Y_pca)
    #     return Y_rgb  # shape: (n_samples, 3)
    def compute_colors_from_multilabels(self,Y):
        """
        输入：Y 是 (n_samples, n_labels) 的多标签 0/1 矩阵
        输出：每个样本对应的 RGB 颜色，归一化在 [0, 1]，更易区分
        """
        from sklearn.preprocessing import MinMaxScaler
        from matplotlib import cm
        import numpy as np

        pca = PCA(n_components=3)
        Y_pca = pca.fit_transform(Y)
        # 非线性拉伸
        Y_pca = np.tanh(Y_pca)
        Y_rgb = MinMaxScaler().fit_transform(Y_pca)
        # 用更鲜明的colormap映射
        # cmap = cm.get_cmap('Spectral')
        # Y_rgb = cmap(Y_rgb)[:, :3]  # 只取RGB，不要alpha通道
        return Y_rgb  # shape: (n_samples, 3)
    
    def force_directed_graph_plot(self, features_np, labels_np, reduced_proto, save_path, top_k=2, use_softmax=True):
        import networkx as nx
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.special import softmax

        # 确保 numpy 格式
        features_np = np.array(features_np)
        labels_np = np.array(labels_np)

        # # 如果是字符串标签（多标签编码后的），先转成数字 ID
        # if labels_np.dtype.type is np.str_:
        #     unique_labels = sorted(set(labels_np))
        #     label_to_id = {label: i for i, label in enumerate(unique_labels)}
        #     numeric_labels = np.array([label_to_id[lbl] for lbl in labels_np])
        # else:
        #     # 如果是 one-hot 多标签，转 argmax；如果是单标签就直接用
        #     if labels_np.ndim > 1:
        #         node_rgb_colors = self.compute_colors_from_multilabels(labels_np)
        #         numeric_labels = np.argmax(labels_np, axis=1)
        #     else:
        #         numeric_labels = labels_np

        # 如果有 TTA 导致 feature 多于标签，尝试聚合
        if features_np.shape[0] > labels_np.shape[0]:
            tta_times = features_np.shape[0] // labels_np.shape[0]
            assert features_np.shape[0] % labels_np.shape[0] == 0, "features 数量无法整除 labels"
            features_np = features_np.reshape(labels_np.shape[0], tta_times, -1).mean(axis=1)

        # 限制样本数量
        sample_limit = 2000
        if features_np.shape[0] > sample_limit:
            sample_indices = np.random.choice(features_np.shape[0], sample_limit, replace=False)
            features_np = features_np[sample_indices]
            labels_np = labels_np[sample_indices]
        
        if use_softmax:
            features_np = softmax(features_np, axis=1)
        else:
            features_np = np.maximum(0, features_np)  # 裁剪负值

         # 计算 RGB 颜色（基于多标签 PCA）
        node_rgb_colors = self.compute_colors_from_multilabels(labels_np)
        node_rgb_colors = np.clip(node_rgb_colors, 0, 1)
        num_centers = features_np.shape[1]
        G = nx.Graph()

        # 添加中心点
        for c in range(num_centers):
            G.add_node(f"C{c}", type='center', weight=0)

        # 添加样本点和边
        for i in range(features_np.shape[0]):
            sample_node = f"S{i}"
            G.add_node(sample_node, type='sample')
            row = features_np[i]
            top_indices = row.argsort()[-top_k:]
            for c in top_indices:
                weight = row[c]
                G.add_edge(sample_node, f"C{c}", weight=weight)
                # 更新该中心点的权重（连接的权重之和）
                G.nodes[f"C{c}"]['weight'] += weight

        # 可视化
        # pos = nx.spring_layout(G)  # 使用 kamada_kawai_layout，可能会更好
        pos = nx.spring_layout(G, seed=42, center=(0, 0), scale=1.0)
         # 准备颜色与大小
        sample_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "sample"]
        center_nodes = [node for node, attr in G.nodes(data=True) if attr["type"] == "center"]
        node_colors = node_rgb_colors[:len(sample_nodes)]
        # cmap = plt.cm.get_cmap("tab20", np.max(numeric_labels) + 1)
        center_sizes = [20 * G.nodes[n]['weight'] for n in center_nodes]

        # # 绘图
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        x_margin = (max(x_vals) - min(x_vals)) * 0.1
        y_margin = (max(y_vals) - min(y_vals)) * 0.1
        plt.figure(figsize=(14, 12))
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=sample_nodes,
            node_color=node_colors,
            node_size=30,
            alpha=0.8
        )
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=center_nodes,
            node_color="red",
            node_size=center_sizes,
            alpha=0.8
        )
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
        plt.xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        plt.ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)
        plt.title("Force-Directed Graph: Soft Assignment (Samples → Confounders)", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    def _hook_fn(self, module, input, output):
        # if self.curr_view_idx % self.num_views != 0:
        #     self.curr_view_idx += 1
        #     return
        pooling = self.model.head.swav_input_proj.pooling
        output=self.model.head.swav_input_proj(output)
        output = self.model.head.head_feature(output)[1]
        # Apply the pooling operation
        pooled_features = pooling(output)
        self.features.append(pooled_features.detach().cpu())
        self.curr_view_idx += 1
        # self.features.append(output[0].detach().cpu())

    def _reduce(self, features):
        if self.method == 'tsne':
            reducer = TSNE(n_components=self.n_components,
                           perplexity=self.perplexity,
                           random_state=self.random_state)
        else:
            reducer = PCA(n_components=self.n_components)
        return reducer.fit_transform(features)

    def _plot(self, features, labels, prototypes, save_path=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        unique_labels = list(set(labels))
        palette = sns.color_palette("hsv", len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot features
        for label in unique_labels:
            idx = [i for i, l in enumerate(labels) if l == label]
            points = features[idx]
            ax.scatter(points[:, 0], points[:, 1], label=label, c=[color_map[label]] * len(idx), s=10)

        # Plot prototypes
        if prototypes is not None:
            ax.scatter(prototypes[:, 0], prototypes[:, 1], c='black', marker='X', s=100, label='Prototypes')

            # Draw lines from each point to its closest prototype
            for feat in features:
                # Compute distance to all prototypes
                distances = np.linalg.norm(prototypes - feat, axis=1)
                closest_proto_idx = np.argmin(distances)
                closest_proto = prototypes[closest_proto_idx]
                ax.plot([feat[0], closest_proto[0]], [feat[1], closest_proto[1]], 'k-', linewidth=0.3, alpha=0.3)

        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _get_layer(self, model, name):
        for attr in name.split('.'):
            model = getattr(model, attr)
        return model
