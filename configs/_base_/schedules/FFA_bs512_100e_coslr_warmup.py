# optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=3e-3, weight_decay=1e-4),
#     clip_grad=dict(max_norm=1.0))
# optim_wrapper = dict(
#     optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4))
# learning policy
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001))
warmup_epochs = 10  # about 10000 iterations for ImageNet-1k
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
        T_max=90,
        # eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs,
        end=100,)
]
# param_scheduler = [
#     # warm up learning rate scheduler
#     dict(
#         type='LinearLR',
#         start_factor=0.1,
#         by_epoch=True,
#         begin=0,
#         end=10,
#         # update by iter
#         # convert_to_iter_based=True,
#     ),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=90,
#         by_epoch=True,
#         begin=10,
#         end=100,
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
#         convert_to_iter_based=True,
#     ),
#     dict(
#         type='MultiStepLR', 
#         by_epoch=True, 
#         milestones=[30, 60, 90], 
#         begin=10,
#         end=100,
#         gamma=0.1)
#     ]
# param_scheduler = [
#     dict(
#         # begin=0,
#         by_epoch=True,
#         div_factor=1e2,
#         total_steps=100,
#         pct_start=0.1,
#         # end=100,
#         eta_max=1e-2,
#         final_div_factor=1e6,
#         type='OneCycleLR'),
# ]

# train, val, test setting

val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
