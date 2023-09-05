# tools/dist_train.sh mask_rcnn_r101_fpn_2x_coco_bwdrat_816x512_cam9.py 4
# python tools/train.py mask_rcnn_r101_fpn_2x_coco_bwdrat_816x512_cam9.py
# python -m lilab.mmdet_dev.convert_mmdet2trt mask_rcnn_r101_fpn_2x_coco_bwdrat_816x512_cam9.py
# 之前黑、灰鼠，专门分割
_base_ = ['configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py']

samples_per_gpu=2
num_classes = 3
checkpoint_config = dict(interval=4)

# load_from = 'work_dirs/mask_rcnn_r101_fpn_2x_coco_3rat_816x512_cam9/latest.pth'
load_from = []

runner = dict(type='EpochBasedRunner', max_epochs=12)
device='cuda'

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='checkpoints/resnet101-5d3b4d8f.pth')),###############
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
    dict(type='Resize', img_scale=(832, 512), keep_ratio=False),
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
        img_scale=(832, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
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

dataset_type = 'CocoDatasetRatBWD'

## the group dataset
data = dict(samples_per_gpu=samples_per_gpu,
            train=dict(type = 'ConcatDataset', 
                datasets = [
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bwd_rat_800x600_0304_trainval.json',
                    #     img_prefix='data/rats/bwd_rat_800x600_0304/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230525_LTZ_trainval_as_bwd.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230525_LTZ/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230524_trainval_as_bwd.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230524/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bwd_rat_1280x800_20230615_trainval.json',
                    #     img_prefix='data/rats/bwd_rat_1280x800_20230615/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bwd_rat_1280x800_20230702_MMF_trainval.json',
                        img_prefix='data/rats/bwd_rat_1280x800_20230702_MMF/')
                    ]), 
            val=dict(type = 'ConcatDataset', 
                datasets = [
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bwd_rat_1280x800_20230615_trainval.json',
                        img_prefix='data/rats/bwd_rat_1280x800_20230615/'),
                        ]), 
            test=dict(type = 'ConcatDataset', 
                pipeline = test_pipeline,
                datasets = [
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bwd_rat_1280x800_20230615_trainval.json',
                        img_prefix='data/rats/bwd_rat_1280x800_20230615/')]))
