# conda activate mmpose
# tools/dist_train.sh mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9.py 4
# python tools/train.py mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9.py
# python -m lilab.mmdet_dev.convert_mmdet2trt mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9.py
_base_ = ['configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py']

samples_per_gpu=2
num_classes = 2
checkpoint_config = dict(interval=4)
# load_from = 'work_dirs/mask_rcnn_r101_fpn_2x_coco_bwrat/latest.pth'
load_from = 'work_dirs/mask_rcnn_r101_fpn_2x_coco_bwrat_816x512_cam9/latest.pth'
runner = dict(type='EpochBasedRunner', max_epochs=8)
device='cuda'

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

dataset_type = 'CocoDatasetRat'

## the black white dataset
data = dict(samples_per_gpu=samples_per_gpu,
            train=dict(type = 'ConcatDataset', 
                datasets = [
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_1126headstage_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_1126headstage/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_230506_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_230506/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800x9_20230209_VPA_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800x9_20230209_VPA/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230524_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230524/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230525_LTZ_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230525_LTZ/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230625_WT_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230625_WT/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230625_WT_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230625_WT/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230724_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230724/'),
                    # dict(pipeline=train_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_20230724_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_20230724/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800x9_20221013_small_trainval.json',
                        img_prefix='data/rats/bw_rat_1280x800x9_20221013_small/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800x9_20221012_big_trainval.json',
                        img_prefix='data/rats/bw_rat_1280x800x9_20221012_big/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800_20230810_LTZ_trainval.json',
                        img_prefix='data/rats/bw_rat_1280x800_20230810_LTZ/'),
                    ]), 
            val=dict(type = 'ConcatDataset', 
                datasets = [
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/coco_BWsiderat_val.json',
                    #     img_prefix='data/rats/bwsiderat/'),
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/coco_bwsiderat800x600_val.json',
                    #     img_prefix='data/rats/bwsiderat800x600/'),
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_800x600_1130_val.json',
                    #     img_prefix='data/rats/bw_rat_800x600_1130/'),
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_800x600_0118_cross_val.json',
                    #     img_prefix='data/rats/bw_rat_800x600_0118_cross/'),
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_0417_val.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_0417/'),
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_0614_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_0614/'),
                    # dict(pipeline=test_pipeline,
                    #     type=dataset_type,
                    #     ann_file='data/rats/bw_rat_1280x800_230506_trainval.json',
                    #     img_prefix='data/rats/bw_rat_1280x800_230506/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_1280x800_20230524_trainval.json',
                        img_prefix='data/rats/bw_rat_1280x800_20230524/'),
                        ]), 
            test=dict(type = 'ConcatDataset', 
                pipeline = test_pipeline,
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
