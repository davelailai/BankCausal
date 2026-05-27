_base_ = [
    # './backbone.py',
    '../_base_/datasets/ChexPert.py',
    '../_base_/schedules/Chest_bs512_16e_coslr_warmup.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs/ChestXPert_BankCausal'

hidden_dim = 512
num_classes=13
prototype_ratio = 2

model = dict(
    warmup_epoch = 5,
    type='MultiLabelImageClassifier',
     backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            checkpoint='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220825-5b3fc7fc.pth',
            # 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
            type='Pretrained'),),
    head=dict(
        num_classes=num_classes,
        in_channels=2048,
        type='Multilabel_BankCausal',
        position_embedding='sine',
        enc_layers=1,
        dec_layers=2,
        # dim_feedforward=256,
        hidden_dim=hidden_dim,
        nheads=8,
        img_size=224,
        patch_size=32,
        with_global=True,
        confounder_norm=True,
        # cals_pre = 'Linear',
        loss=dict(type='AsymmetricLoss', loss_weight=1.0, use_sigmoid=True),
        causal_type = 'explicit',
        swav_loss = dict(
                type='MultiLabelSwAVLoss',
                feat_dim=hidden_dim,  # equal to neck['out_channels']
                epsilon=0.05,
                temperature=0.1,
                num_crops=[2],
                num_prototypes=num_classes*prototype_ratio,
                crops_for_assign=[0,1],
                # weight = 1.0
            )
    )
)

# param_scheduler = [
#     # Warm-up learning rate scheduler for 10 epochs
#     dict(
#         type='LinearLR',
#         start_factor=0.01,
#         by_epoch=True,
#         end=10,  # Warm-up phase for 10 epochs
#     ),
#     # Constant learning rate phase for 30 epochs
#     dict(
#         type='ConstantLR',
#         factor=1.0,  # Keep LR constant
#         by_epoch=True,
#         begin=10,  # Starts after warm-up
#         end=30,  # Runs for 30 epochs (10-40)
#     ),
#     # Cosine annealing learning rate for 90 epochs
#     dict(
#         type='CosineAnnealingLR',
#         T_max=90,  # Cosine decay over 90 epochs
#         by_epoch=True,
#         begin=30,  # Starts after constant LR phase
#         end=120,  # Total epochs (10 warm-up + 20 constant + 90 cosine)
#     )
# ]
train_cfg = dict(by_epoch=True, max_epochs=40, val_interval=1)
# train_cfg = dict(type='MultiLabelEpochWarmUpTrainLoop', max_epochs=100, val_interval=1)
test_cfg = dict(type='MultiLabelWarmUpTestLoop')
# val_cfg = dict(type='MultiLabelWarmUpValLoop')
# model_wrapper_cfg = dict(
#                         type='MultiLabel_MMGANDistributedDataParallel',
#                         find_unused_parameters=True,
#                         # detect_anomalous_params=True
#                          )
default_hooks = dict(
    # record the time of every iteration.
    logger=dict(type='LoggerHook', interval=50),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3)
)
model_wrapper_cfg=dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=True,
        # detect_anomalous_params=True
        )
custom_hooks = [
    dict(
        type='MultiLabelSwAVHook',
        priority='VERY_HIGH',
        batch_size=32,
        epoch_queue_starts=0,
        # crops_for_assign=[0],
        feat_dim=hidden_dim,
        queue_length=10000,
        frozen_epoch_cfg=dict(prototypes=15),
        loss_decay = dict(
            start_weight=1,
            end_weight=0,    # Final weight
            decay_type='linear',  
            start_epoch=5, # or 'exponential'
            total_epochs=15#   # Should match max_epochs)),
        )
    ),
    dict(
        type='PrototypeClusteringHook',
        save_path=f'{work_dir}/Clustering.png',  # Path to SwAV loss weight
        proto_attr='head.swav_loss_module.prototypes'
    )
]