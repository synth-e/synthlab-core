from .base import AtomicType
from .index import IndexedFile

import numpy as np
from io import BytesIO
import requests
import torch
from PIL import Image
import base64
from copy import deepcopy
from synthlab_core.utilities.misc import decltype
from hashlib import sha512
import cv2
import os
import json
import typing
from .image import ImageWrapper
from synthlab_core.utilities.data import DataContext
from synthlab_core.utilities.misc.visualize import apply_mask_to_image, generate_colors



class MaskWrapper(AtomicType):
    # @TODO: https://numpy.org/doc/stable/reference/generated/numpy.savez.html
    # impl to store maks with labels/palettes in the same file
    """MaskWrapper: A wrapper for mask, it can be used to load mask from different sources, format and highly compatible with the project codebase.

    Here is an example of how to use it:

    from segmentation_data_generator.common.atomic import MaskWrapper, ImageWrapper
    from segmentation_data_generator.utilities.graph_algo.travelsal import mask2boxes
    from segmentation_data_generator.cv.segmentation.ClipSegSegmentor

    img = ImageWrapper('path2file.png')
    clipseg = ClipSegSegmentor(mask_threshold=.5, uncertain_threshold=.4)

    mask: MaskWrapper = clipseg(img, TextualPrompt(labels = ["dog", "cat"]))

    print(len(mask)) # the number of layers in the mask, including background and uncertain layers

    for layer in mask:
        print(layer.labels[1]) # here layer is a Binary MaskWrapper, 1 is stand for obj and 0 is stand for background
        layer.render(img, opacity = 0.5).save(layer.labels[1] + '.png')

    # or do some post process to the mask, we can convert it to numpy array
    mask_npy = mask.numpy

    # or we can convert it to tensor
    mask_tensor = mask.tensor

    # after completed processing, we can create again a new MaskWrapper
    processed_mask = MaskWrapper(mask_npy, mask.labels) # the labels could be changed if needed
    """

    BACKGROUND_LABEL = "background"
    UNCERTAIN_LABEL = "uncertain"
    UNKNOWN = "unknown"

    BACKGROUND_PALLETE = [0, 0, 0]
    UNCERTAIN_PALLETE = [255, 255, 255]

    UNCERTAIN_ID = 255
    BACKGROUND_ID = 0

    @classmethod
    def from_file(cls, p):
        if p.endswith(".npy"):
            return cls(np.load(p))

        if p.endswith(".pt"):
            return cls(torch.load(p))

        if p.startswith("http"):
            return cls(np.array(Image.open(BytesIO(requests.get(p).content))))

        if p.endswith(".png") or p.endswith(".jpg"):
            return cls(np.array(Image.open(p)))

    @classmethod
    def from_buffer(cls, b: bytes):
        json_data = json.loads(b.decode())
        return cls(json_data["regions"], json_data["labels"], json_data["palletes"])

    def to_buffer(self):
        return json.dumps(self.json).encode()

    def to_web_compatible(self):
        white_img = np.ones((512, 512, 3), np.uint8) * 255

        image_mask = apply_mask_to_image(
            white_img, self.numpy, self.labels, self.palletes, 1.0
        )

        return ImageWrapper(image_mask).to_web_compatible()

    def hash(self):
        data = self.to_web_compatible()
        return sha512(data.encode()).hexdigest()

    @classmethod
    def load_mask(cls, mask):
        if isinstance(mask, list):
            return np.array(mask, np.uint8)

        if isinstance(mask, IndexedFile):
            return cls.load_mask(mask.path)

        if isinstance(mask, np.ndarray):
            return mask.astype(np.uint8)

        if isinstance(mask, MaskWrapper):
            return mask.numpy.astype(np.uint8)

        if isinstance(mask, torch.Tensor):
            return mask.cpu().numpy().astype(np.uint8)

        if isinstance(mask, str):
            buffer = None
            if mask.startswith("http"):
                buffer = requests.get(mask).content

            elif mask.startswith("data:image/png;base64,") or mask.startswith(
                "data:image/jpg;base64,"
            ):
                buffer = base64.b64decode(mask.split(",")[1])

            elif os.path.exists(mask):
                if mask.endswith(".npy"):
                    return np.load(mask)

                with open(mask, "rb") as f:
                    buffer = f.read()

            if buffer is not None:
                if decltype.is_image(buffer):
                    return np.array(Image.open(BytesIO(buffer))).astype(np.uint8)

                if decltype.is_nparray(buffer):
                    return np.load(buffer).astype(np.uint8)

        raise ValueError(f"Unsupported image type: {type(mask)}")

    def __init__(self, mask, labels: typing.Union[dict, list] = None, palletes: dict = None):
        """MaskWrapper: A wrapper for mask, it can be used to load mask from different sources, format and highly compatible with the project codebase.
        mask: np.ndarray, list, str, torch.Tensor, ImageWrapper, MaskWrapper
        labels: dict[uint8, str], optional, the labels for the mask, the key is the id and the value is the label, default is None

        For example, if I have a 2x2 mask, I can create a mask wrapper like:

        ```python
        mask = MaskWrapper(
            mask = [[1, 2],
                    [3, 4]],
            labels = {
                1: "top left",
                2: "top right",
                3: "bottom left",
                4: "bottom right"
            }
        )
        ```

        Note that if the labels is not provided on the initialization,
        - 0 id will be marked as background as default
        - 255 id will be marked as uncertain as default
        - other ids will be marked as unknown as default
        """

        self.mask = self.load_mask(mask)

        assert (
            len(self.mask.shape) == 2
        ), f"The mask must be a 2D array. Got {self.mask.shape}"

        if labels is None:
            labels = {}

        if palletes is None:
            palletes = {}

        self.labels = None
        self.palletes = None

        self.set_labels(labels)
        self.set_palletes(palletes)

        self.w, self.h = self.mask.shape

        for un in np.unique(self.mask):
            if un not in self.labels.keys():
                self.labels[un] = self.UNKNOWN

    def is_binary(self):
        return len(np.unique(self.mask)) <= 2

    def clone(self):
        return deepcopy(self)

    @property
    def numpy(self) -> np.ndarray:
        return self.mask

    @property
    def tensor(self):
        return torch.from_numpy(self.numpy)

    @property
    def pil(self):
        return Image.fromarray(self.numpy)

    def points(self, value: int = 1):
        return np.argwhere(self.mask == value)

    def __len__(self):
        return len(np.unique(self.mask))

    def __getitem__(self, index):
        return self.mask == index

    @property
    def shape(self):
        return self.mask.shape

    def __str__(self) -> str:
        return f"MaskWrapper({self.labels}, {self.mask.shape})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def json(self):
        return {
            "labels": {int(id): label for id, label in self.labels.items()},
            "palletes": {
                int(id): (color.tolist() if isinstance(color, np.ndarray) else color)
                for id, color in self.palletes.items()
            },
            "regions": self.mask.tolist(),
        }

    def __iter__(self) -> iter:
        for id, label in self.labels.items():
            if id in [self.BACKGROUND_ID]:
                continue

            yield MaskWrapper(
                (self.mask == id).astype(np.uint8),
                {
                    1: label,
                    self.BACKGROUND_ID: self.BACKGROUND_LABEL,
                },
            )

    # render the image with all masks
    def visualize(
        self, image: ImageWrapper, opacity=0.55, *args, **kwargs
    ) -> ImageWrapper:
        image_mask = apply_mask_to_image(
            image.numpy, self.numpy, self.labels, self.palletes, opacity
        )
        return ImageWrapper(image_mask)

    def resize(self, size) -> "MaskWrapper":
        return MaskWrapper(
            np.array(Image.fromarray(self.numpy).resize(size, Image.NEAREST)),
            self.labels,
        )

    def render_heatmap(self, image: ImageWrapper):
        # TODO: @minhtuan, move this to utilities.misc.visualize
        img_cv = cv2.cvtColor(image.numpy, cv2.COLOR_RGB2BGR)
        heat_map = cv2.applyColorMap(self.mask, cv2.COLORMAP_JET)

        viz = 0.4 * img_cv + 0.6 * heat_map
        viz = cv2.cvtColor(viz.astype("uint8"), cv2.COLOR_BGR2RGB)
        return ImageWrapper(viz)

    def set_palletes(self, palletes: typing.Union[dict, list]):
        if isinstance(palletes, list):
            palletes = {id: color for id, color in enumerate(palletes)}

        palletes.setdefault(self.BACKGROUND_ID, self.BACKGROUND_PALLETE)
        palletes.setdefault(self.UNCERTAIN_ID, self.UNCERTAIN_PALLETE)

        if not isinstance(palletes, dict):
            raise ValueError(f"Palletes must be a dictionary, got {type(palletes)}")
        self.palletes = palletes

    def set_labels(self, labels: typing.Union[dict, list]):
        if isinstance(labels, list):
            labels = {id: label for id, label in enumerate(labels)}

        labels.setdefault(self.BACKGROUND_ID, self.BACKGROUND_LABEL)
        labels.setdefault(self.UNCERTAIN_ID, self.UNCERTAIN_LABEL)

        if not isinstance(labels, dict):
            raise ValueError(f"Labels must be a dictionary, got {type(labels)}")

        self.labels = labels

    def remap_labels(self, labels: typing.Union[dict, list]):
        """Remap the labels to the current mask

        Example:

        image = ImageWrapper('path2file.png')
        txt_prompt = TextualPrompt(labels=['dog', 'cat'])
        bbox: BBoxListWrapper = dino(image, txt_prompt)
        mask: MaskWrapper = sam(image, bbox)

        # somewhere else
        mask.set_labels(COCO_LABEL_DICT)
        save(mask)

        """

        if isinstance(labels, list):
            labels = {id: label for id, label in enumerate(labels)}

        if not isinstance(labels, dict):
            raise ValueError(f"Labels must be a dictionary, got {type(labels)}")

        reverse_current_labels = {e: [] for e in self.labels.values()}

        for id, label in self.labels.items():
            reverse_current_labels[label].append(id)

        new_mask = np.zeros_like(self.mask)

        missing_values, dropped_labels = [], []
        for id, label in labels.items():
            if label in reverse_current_labels:
                for current_id in reverse_current_labels[label]:
                    new_mask[self.mask == current_id] = id

            else:
                missing_values.append(label)

        for k in reverse_current_labels:
            if k not in labels.values():
                dropped_labels.append(k)

        self.mask = new_mask
        self.labels = labels

    def clone(self) -> "MaskWrapper":
        return MaskWrapper(np.array(self.mask, copy=True), deepcopy(self.labels))

    def to_data(self, data: DataContext, remap=True) -> "MaskWrapper":
        if not isinstance(data, DataContext):
            raise ValueError(f"data must be DataContext, got {type(data)}")

        new_mask_wrapper = self.clone()

        new_mask_wrapper.set_palletes(data.id2palette)
        if remap:
            new_mask_wrapper.remap_labels(data.id2label)
        else:
            new_mask_wrapper.set_labels(data.id2label)

        return new_mask_wrapper