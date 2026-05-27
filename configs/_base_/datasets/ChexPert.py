dataset_type ='MultiLabel_Ffa'
data_preprocessor = dict(
    num_classes=13,
    # RGB format normalization parameters
    mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    # mean=[0.5],
    # std=[0.5],
    # convert image from BGR to RGB
    to_rgb=True,
)
color_distort_strength = 0.2  # reduced for medical images
view_pipeline1 = [
    dict(
        type='RandomResizedCrop',
        scale=(224, 224),
        crop_ratio_range=(0.8, 1.0),
        aspect_ratio_range=(0.75, 1.33)
    ),

    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.2 * color_distort_strength,
                contrast=0.2 * color_distort_strength,
                saturation=0.0,  # grayscale images
                hue=0.0
            )
        ],
        prob=0.5
    ),
    dict(
        type='RandomGrayscale',
        prob=0.1,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)
    ),
    dict(
        type='GaussianBlur',
        magnitude_range=(0.1, 0.3),
        prob=0.1
    ),
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


train_dataloader = dict(
    batch_size=32,
    num_workers=5,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # collate_fn=dict(type='custom_collate'),
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/Chexpert/mmpretrain_train.pkl',
        pipeline=train_pipeline),
)

val_dataloader = dict(
    batch_size=128,
    num_workers=5,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # collate_fn=dict(type='custom_collate'),
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/Chexpert/mmpretrain_val.pkl',
        pipeline=test_pipeline),
)


test_dataloader = dict(
    batch_size=128,
    num_workers=5,
    sampler=dict(type='DefaultSampler', shuffle=True),
    # collate_fn=dict(type='custom_collate'),
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/Chexpert/mmpretrain_test.pkl',
        pipeline=test_pipeline)
)

# If you want standard test, please manually configure the test dataset
val_evaluator = [
    # dict(type='RankingMultiLabelMetric',
    #     items=['f1-score'],
    #     # average='both',
    #     # Ranking=Ranking,
    #     thr=0.5),
    # dict(type='RankingROC_AUC'),
    dict(type='RankingAverageAUC'),
    dict(type='MultiLabel_AveragePrecision'), 
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
    dict(type='MultiLabel_AveragePrecision'), 
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