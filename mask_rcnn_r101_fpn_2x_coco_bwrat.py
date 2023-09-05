# tools/dist_train.sh mask_rcnn_r101_fpn_2x_coco_bwrat.py 4

_base_ = ['configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py']

samples_per_gpu=2
num_classes = 2
checkpoint_config = dict(interval=4)
load_from = 'work_dirs/mask_rcnn_r101_fpn_2x_coco_bwrat/latest.pth'
runner = dict(type='EpochBasedRunner', max_epochs=12)

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='checkpoints/resnet101-5d3b4d8f.pth')),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes),
        mask_head=dict(
            num_classes=num_classes),
    test_cfg=dict(
        rpn=dict(
            nms_pre=200,
            max_per_img=200,
        ),
        rcnn=dict(
            max_per_img=20,
        )
    )
))


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'CocoDatasetRat'

# the black white dataset
data = dict(samples_per_gpu=samples_per_gpu,
            train=dict(type = 'ConcatDataset', 
                datasets = [
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/coco_BWsiderat_train.json',
                    #     img_prefix='data/rats/bwsiderat/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_bwsiderat800x600_train.json',
                        img_prefix='data/rats/bwsiderat800x600/'), 
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_train.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_0118_cross_train.json',
                        img_prefix='data/rats/bw_rat_800x600_0118_cross/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_0224_play_train.json',
                        img_prefix='data/rats/bw_rat_800x600_0224_play/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bwd_rat_800x600_0304_trainval_as_bw.json',
                    #     img_prefix='data/rats/bwd_rat_800x600_0304/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800_0417_train.json',
                        img_prefix='data/rats/bw_rat_1280x800_0417/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800_0424_train.json',
                        img_prefix='data/rats/bw_rat_1280x800_0424/'),
                        ]), 
            val=dict(type = 'ConcatDataset', 
                datasets = [
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/coco_BWsiderat_val.json',
                    #     img_prefix='data/rats/bwsiderat/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_bwsiderat800x600_val.json',
                        img_prefix='data/rats/bwsiderat800x600/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_val.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/'),
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_800x600_0118_cross_val.json',
                    #     img_prefix='data/rats/bw_rat_800x600_0118_cross/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800_0417_val.json',
                        img_prefix='data/rats/bw_rat_1280x800_0417/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800_0424_val.json',
                        img_prefix='data/rats/bw_rat_1280x800_0424/'),
                        ]), 
            test=dict(type = 'ConcatDataset', 
                datasets = [
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_bwsiderat800x600_val.json',
                        img_prefix='data/rats/bwsiderat800x600/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_val.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_0118_cross_val.json',
                        img_prefix='data/rats/bw_rat_800x600_0118_cross/')]))

## the black white dataset, + bwd dataset
# data = dict(samples_per_gpu=samples_per_gpu,
#             train=dict(pipeline=train_pipeline,
#                         type=dataset_type,
#                         ann_file='data/rats/coco_BWsiderat_train.json',
#                         img_prefix='data/rats/bwsiderat/'), 
#             val=dict(pipeline=test_pipeline,
#                         type=dataset_type,
#                         ann_file='data/rats/coco_bwsiderat800x600_val.json',
#                         img_prefix='data/rats/bwsiderat800x600/'), 
#             test=dict(pipeline=test_pipeline,
#                         type=dataset_type,
#                         ann_file='data/rats/coco_bwsiderat800x600_val.json',
#                         img_prefix='data/rats/bwsiderat800x600/'))