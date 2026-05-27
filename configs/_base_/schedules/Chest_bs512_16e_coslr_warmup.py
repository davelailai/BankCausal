# # optimizer
# optim_wrapper = dict(
    # optimizer=dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001))

# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-5)
# )

# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.01, by_epoch=True, end=10),
#     dict(
#         type='ReduceLROnPlateau',
#         monitor='multi-label/mAUC',
#         factor=0.5,
#         patience=2,
#         min_lr=1e-6,
#         mode='max',
#         verbose=True
#     )
# ]
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001))
warmup_epochs = 5  # about 10000 iterations for ImageNet-1k
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        # convert_to_iter_based=True/
        ),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        # T_max=90,
        # eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs)
]

# optim_wrapper = dict(
#     optimizer=dict(lr=5e-05, type='AdamW', weight_decay=1e-05))
# param_scheduler = [
#     dict(by_epoch=True, end=10, start_factor=0.01, type='LinearLR'),
#     dict(
#         begin=10,
#         cooldown=1,
#         factor=0.5,
#         min_value=1e-06,
#         monitor='multi-label/mAUC',
#         param_name='lr',
#         patience=2,
#         rule='greater',
#         threshold=0.001,
#         threshold_rule='rel',
#         type='ReduceOnPlateauParamScheduler'),
# ]

# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-5)
# )

# param_scheduler = [
#     # 5个epoch线性warmup
#     dict(type='LinearLR', start_factor=0.01, by_epoch=True, end=5),
#     # 余弦退火到16 epoch
#     dict(
#         type='CosineAnnealingLR',
#         T_max=11,  # 总epoch数-预热epoch数
#         eta_min=1e-6,
#         by_epoch=True,
#         begin=5,
#         end=16
#     )
# ]
# train_cfg = dict(by_epoch=True, max_epochs=16, val_interval=1)

# # learning policy
# warmup_epochs = 5  # about 10000 iterations for ImageNet-1k
# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=0.01,
#         by_epoch=True,
#         end=warmup_epochs,
#         # update by iter
#         # convert_to_iter_based=True/
#         ),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=20,
#         eta_min=1e-5,
#         by_epoch=True,
#         begin=warmup_epochs,
#         end=40)
# ]

# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-5)
# )

# param_scheduler = [
#     dict(type='LinearLR', start_factor=0.01, by_epoch=True, end=10),  # longer warm-up
#     dict(
#         type='CosineAnnealingLR',
#         T_max=30,  # total epochs - warmup
#         eta_min=1e-6,
#         by_epoch=True,
#         begin=10,
#         end=40
#     )
# ]
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         start_factor=0.1,
#         by_epoch=True,
#         begin=0,
#         end=10,
#         # update by iter
#         # convert_to_iter_based=True,
#     ),
#     dict(
#         type='MultiStepLR', 
#         by_epoch=True, 
#         milestones=[30, 60, 90], 
#         begin=10,
#         end=100,
#         gamma=0.1)
#     ]

# train, val, test setting

val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
