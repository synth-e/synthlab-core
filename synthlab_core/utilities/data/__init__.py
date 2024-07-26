from .label import VOC2012_LABEL_DICT
from .pallete import VOC2012_PALETTE


class DataContext:

    def __init__(self, id2label: dict, id2palette: dict) -> None:
        self.id2label = id2label
        self.id2palette = id2palette


VOC2012_DATA_CONTEXT = DataContext(VOC2012_LABEL_DICT, VOC2012_PALETTE)
