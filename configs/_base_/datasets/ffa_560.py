dataset_type ='MultiLabel_Ffa'
data_preprocessor = dict(
    # type='MultiLabelSelfSupDataPreprocessor',
    # num_classes=7,
    # RGB format normalization parameters
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # mean=[0.5],
    # std=[0.5],
    # convert image from BGR to RGB
    to_rgb=True,
)
# color_distort_strength = 1.0
view_pipeline1 = [
    dict(type='Resize',scale=512),
    # dict(
    #     type='RandomResizedCrop',
    #     scale=448,
    #     crop_ratio_range=(0.9, 1.0),  # 40% to 100% of area, preserves context
    #     aspect_ratio_range=(1.0, 1.0),  # matches UWF aspect ratio
    #     # crop_ratio_range=(1.0, 1.0),
    #     backend='pillow',
    #     interpolation='bicubic'),
    dict(type='RandomCrop',crop_size=448),
    # dict(
    #     type='RandomApply',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.8 * color_distort_strength,
    #             contrast=0.8 * color_distort_strength,
    #             saturation=0.8 * color_distort_strength,
    #             hue=0.2 * color_distort_strength)
    #     ],
    #     prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='GaussianBlur', magnitude_range=(0.1, 1.0), prob=0.2),  
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),   
]



train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize',scale=512),
    dict(
        type='MultiLabelMultiView',
        num_views=2,
        transforms=[view_pipeline1]),
    # dict(type='RandomCrop',crop_size=448),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize',scale=512),
    dict(
        type='MultiLabelMultiView',
        num_views=[1],
        transforms=[view_pipeline1]),
    # dict(type='RandomCrop',crop_size=448),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]


# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize',scale=512),
#     dict(type='CenterCrop',crop_size=448),
#     dict(type='PackInputs'),
# ]

tta_model = dict(type='AverageClsScoreTTA')
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize',scale=512),
    dict(
        type='TestTimeAug',
        transforms=[
                [dict(type='MultiLabelMultiView',num_views=[1],transforms=[view_pipeline1])]*10,
                # [dict(type='RandomFlip', prob=0.5, direction='horizontal')]*2,
                [test_pipeline[-1]]
                ]
      ),
]
# tta_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=512),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='RandomResizedCrop', scale=448, crop_ratio_range=(0.9, 1.0), aspect_ratio_range=(1.0, 1.0), backend='pillow', interpolation='bicubic'),
#                 dict(type='RandomFlip', prob=0.5, direction='horizontal')
#             ] * 10,
#             [dict(type='PackInputs')]
#         ]
#     ),
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='ResizeEdge',
#         scale=560,
#         edge='short',
#         backend='pillow',
#         interpolation='bicubic'),
#     dict(type='CenterCrop', crop_size=560),
#     dict(type='PackInputs'),
# ]
folder='data/FFA'
train_dataloader = dict(
    batch_size= 32,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/FFA/train_class6.pkl',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/FFA/val_class6.pkl',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)


test_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/FFA/test_class6.pkl',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Ranking = False
# If you want standard test, please manually configure the test dataset
val_evaluator = [
    # dict(type='RankingMultiLabelMetric',
    #     items=['f1-score'],
    #     # average='both',
    #     # Ranking=Ranking,
    #     thr=0.5),
    dict(type='RankingROC_AUC'),
    dict(type='RankingAverageAUC'),
    dict(type='AveragePrecision'), 
    dict(average=None, type='RankingAverageAUC'),
  
    dict(type='RankingMultiLabelMetric',
        items=['precision', 'recall', 'f1-score','support'],
        # average='both',
        # Ranking=Ranking,
        thr=0.5),
    dict(type='RankingMultiLabelMetric',
        items=['precision', 'recall', 'f1-score','support'],
        average='micro',
        # Ranking=Ranking,
        thr=0.5),
    # dict(type='MultiLabelMetric',
    #     items=['precision', 'recall', 'f1-score','support'],
    #     average=None,
    #     thr=0.5),
    # dict(type='ROC_AUC',average=None),
]

test_evaluator = [
    dict(type='RankingROC_AUC'),
    dict(type='RankingAverageAUC'),
    dict(type='AveragePrecision'), 
    dict(average=None, type='RankingAverageAUC'),
  
    dict(type='RankingMultiLabelMetric',
        items=['f1-score', 'precision', 'recall', 'support'],
        # Ranking=Ranking,
        # average='both',
        thr=0.5),
    dict(type='RankingMultiLabelMetric',
        items=['precision', 'recall', 'f1-score','support'],
        # Ranking=Ranking,
        average='micro',
        thr=0.5),
    dict(type='RankingMultiLabelMetric',
        items=['precision', 'recall', 'f1-score','support'],
        # Ranking=Ranking,
        average=None,
        thr=0.5),
    dict(type='RankingROC_AUC',average=None),
]