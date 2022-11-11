from .coco import DATASETS, CocoDataset


@DATASETS.register_module()
class CocoDatasetRat(CocoDataset):
    CLASSES = ('rat_black', 'rat_white')


@DATASETS.register_module()
class CocoDatasetRatOneclass(CocoDataset):
    CLASSES = ('rat',)