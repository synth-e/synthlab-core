VOC2012_CATEGORIES = [
    "background",
    "plane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "monitor",
]

VOC2012_LABEL_DICT = {id: label for id, label in enumerate(VOC2012_CATEGORIES)}
VOC2012_LABEL_DICT[255] = "uncertain"

# TODO: ADD COCO
