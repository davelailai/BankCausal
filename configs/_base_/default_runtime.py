# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=50),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),
    
    ema=dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',  # EMA 策略
        begin_epoch=10,                        # 从第1个epoch开始EMA
        evaluate_on_ema=True,                 # 验证/测试使用EMA模型
        evaluate_on_origin=False,             # 是否验证原始模型
        momentum=0.9997,                      # EMA动量
        priority='NORMAL'),
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', 
                    interval=1,
                    # save_best='multi-label/f1-score',  # Specify the metric for best checkpoint
                    # rule='greater',  # Specify that a higher value is better
                    save_best='auto',
                    max_keep_ckpts=3
                    ),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)


# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
# vis_backends = [dict(type='LocalVisBackend')]
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)


# model_wrapper_cfg=dict(
#         type='MMDistributedDataParallel',
#         find_unused_parameters=True,
#         detect_anomalous_params=True
#         )