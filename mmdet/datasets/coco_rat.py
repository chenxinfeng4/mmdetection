from .coco import DATASETS, CocoDataset


@DATASETS.register_module()
class CocoDatasetRat(CocoDataset):
    CLASSES = ('rat_black', 'rat_white')


@DATASETS.register_module()
class CocoDatasetRatOneclass(CocoDataset):
    CLASSES = ('rat',)

@DATASETS.register_module()
class CocoDatasetRatBWD(CocoDataset):
    CLASSES = ('rat_black', 'rat_white', 'rat_dot')

@DATASETS.register_module()
class CocoDatasetRatBWDG(CocoDataset):
    CLASSES = ('rat_white','rat_gray','rat_black','rat_dot')

@DATASETS.register_module()
class CocoDatasetRatBWDR(CocoDataset):
    CLASSES = ('rat_black','rat_white','rat_dot','rat_row')

@DATASETS.register_module()
class CocoDatasetMar(CocoDataset):
    CLASSES = ('mar')
    