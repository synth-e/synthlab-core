from .base import AtomicType
from .mask import MaskWrapper
from .image import ImageWrapper

import torch
from PIL import Image
import numpy as np
import json
from hashlib import sha512
import cv2
import typing
import os
from io import BytesIO

from synthlab_core.utilities.misc.visualize import generate_colors
from synthlab_core.utilities.graph_algo.travelsal import mask2boxes

class BBoxWrapper(AtomicType):

    @classmethod
    def from_file(cls, p) -> "BBoxWrapper":
        """
        Load the bounding box from a file
        """
        if p.endswith(".npy"):
            return cls(np.load(p))

        if p.endswith(".pt"):
            return cls(torch.load(p))

        if p.endswith(".png") or p.endswith(".jpg"):
            return cls(np.array(Image.open(p)))

        raise ValueError(f"Unsupported file type: {p}")

    @classmethod
    def from_buffer(cls, b: bytes) -> "BBoxWrapper":
        """
        Load the bounding box from a numpy array buffer
        """
        return cls(**json.loads(b.decode()))

    def to_buffer(self) -> bytes:
        return json.dumps(self.json_serializable).encode()

    def to_web_compatible(self):
        white_img = np.zeros((self._baseh or 512, self._basew or 512, 3), np.uint8)

        x1, y1, x2, y2 = self.bbox
        nparray = cv2.rectangle(white_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return ImageWrapper(nparray).to_web_compatible()

    def hash(self):
        _sha512 = sha512()

        for i in self.bbox:
            _sha512.update(i.to_bytes(4, byteorder="big"))

        return _sha512.hexdigest()

    @classmethod
    def load_box(cls, bbox: typing.Any):
        if isinstance(bbox, list):
            # [x1, y1, x2, y2]
            # [top left, bottom right]

            assert (
                len(bbox) == 4 or len(bbox) == 6
            ), f"Bounding box must be a list of 4 or 6 elements, got {len(bbox)}"

            # re-order the list
            x1, y1, x2, y2 = bbox

            if x1 > x2:
                x1, x2 = x2, x1

            if y1 > y2:
                y1, y2 = y2, y1

            if len(bbox) == 4:
                return [x1, y1, x2, y2]

            # return baseh, basew
            return [x1, y1, x2, y2, bbox[4], bbox[5]]

        if isinstance(bbox, BBoxWrapper):
            return bbox.bbox

        if isinstance(bbox, np.ndarray):
            return bbox.tolist()

        if isinstance(bbox, torch.Tensor):
            return bbox.cpu().numpy()

        if isinstance(bbox, str):
            if os.path.exists(bbox) and bbox.endswith(".npy"):
                return cls.load_box(np.load(bbox))

        raise ValueError(f"Unsupported box of type: {type(bbox)}")

    def __init__(
        self, bbox: list, label: str = None, logit=None, baseh=None, basew=None
    ):
        self.bbox = self.load_box(bbox)
        self._label = label
        self._logit = logit

        # to store the base height and width of the image
        self._baseh = baseh
        self._basew = basew

    @property
    def xyxy(self):
        return self.bbox

    @property
    def x1(self):
        return self.bbox[0]

    @property
    def y1(self):
        return self.bbox[1]

    @property
    def x2(self):
        return self.bbox[2]

    @property
    def y2(self):
        return self.bbox[3]

    @property
    def label(self):
        return self._label

    @property
    def logit(self):
        return self._logit or 1.0

    @property
    def json_serializable(self):
        return {
            "bbox": self.bbox,
            "label": self.label,
            "logit": self.logit,
            "baseh": self._baseh,
            "basew": self._basew,
        }

    def aslist(self):
        return self.bbox

    def visualize(
        self, visual_resource: ImageWrapper, color=(0, 255, 0), thickness=2
    ) -> ImageWrapper:
        imgh, imgw = visual_resource.size()
        x1, y1, x2, y2 = self.bbox

        if x1 > imgw:
            x1 = imgw

        if x2 > imgw:
            x2 = imgw

        if y1 > imgh:
            y1 = imgh

        if y2 > imgh:
            y2 = imgh

        # empty
        if (x2 - x1) * (y2 - y1) == 0:
            return visual_resource

        nparray = visual_resource.clone().numpy
        nparray = cv2.rectangle(nparray, (x1, y1), (x2, y2), color, thickness)

        if self.label is not None:
            nparray = cv2.putText(
                nparray,
                f"{self.label} - {self.logit: .2f}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return ImageWrapper(nparray)

    def cutout(self, image: ImageWrapper) -> ImageWrapper:
        npy = image.numpy
        return ImageWrapper(npy[self.y1 : self.y2, self.x1 : self.x2])


class BBoxListWrapper(AtomicType, list):

    @classmethod
    def from_file(cls, p) -> "BBoxWrapper":
        """
        Load the bounding box from a file
        """
        if p.endswith(".npy"):
            return cls(np.load(p))

        if p.endswith(".pt"):
            return cls(torch.load(p))

        if p.endswith(".png") or p.endswith(".jpg"):
            return cls(np.array(Image.open(p)))

        raise ValueError(f"Unsupported file type: {p}")

    @classmethod
    def from_buffer(cls, b: bytes) -> "BBoxWrapper":
        """
        Load the bounding box from a numpy array buffer
        """
        return cls(np.array(np.load(BytesIO(b))))

    def to_buffer(self) -> bytes:
        return json.dumps(self.json_serializable).encode()

    def to_web_compatible(self):
        baseh, basew = self._baseh or 512, self._basew or 512
        white_img = np.zeros((baseh, basew, 3), np.uint8)

        for bbox in self:
            x1, y1, x2, y2 = bbox.bbox

            white_img = cv2.rectangle(
                white_img, (x1, y1), (x2, y2), generate_colors(1)[0].tolist(), 2
            )

            if bbox.label is not None:
                white_img = cv2.putText(
                    white_img,
                    f"{bbox.label}",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return ImageWrapper(white_img).to_web_compatible()

    def hash(self):
        _sha512 = sha512()

        for bbox in self:
            for i in bbox.bbox:
                _sha512.update(i.to_bytes(4, byteorder="big"))

        return _sha512.hexdigest()

    def __init__(
        self, boxes=[], labels: list = [], logits: list = [], baseh=None, basew=None
    ) -> None:
        super().__init__()

        if len(labels) < len(boxes):
            labels.extend([None] * (len(boxes) - len(labels)))

        if len(logits) < len(boxes):
            logits.extend([None] * (len(boxes) - len(logits)))

        for box, label, logit in zip(boxes, labels, logits):
            if not isinstance(box, BBoxWrapper):
                self.append(BBoxWrapper(box, label, logit, baseh=baseh, basew=basew))
            else:
                self.append(box)

        self._baseh = baseh
        self._basew = basew

    @property
    def json_serializable(self):
        return [bbox.json_serializable for bbox in self]

    @staticmethod
    def from_json(json_data: dict):
        if "bbox" in json_data:
            return BBoxListWrapper(
                json_data["bbox"],
                json_data.get("label", []),
                json_data.get("logit", []),
            )

        if isinstance(json_data, list):
            bboxes, labels, logits = [], [], []

            for item in json_data:
                bboxes.append(item["bbox"])
                labels.append(item.get("label", None))
                logits.append(item.get("logit", None))

            return BBoxListWrapper(bboxes, labels, logits)

        raise ValueError(f"Unsupported json data: {json_data}")

    def visualize(
        self, visual_resource: ImageWrapper, color=(0, 255, 0), thickness=2
    ) -> ImageWrapper:
        nparray = visual_resource.clone().numpy

        for bbox in self:
            x1, y1, x2, y2 = bbox.aslist()
            nparray = cv2.rectangle(nparray, (x1, y1), (x2, y2), color, thickness)

            if bbox.label is not None:
                nparray = cv2.putText(
                    nparray,
                    f"{bbox.label} - {bbox.logit: .2f}",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        return ImageWrapper(nparray)

    @staticmethod
    def from_mask(mask: MaskWrapper) -> "BBoxListWrapper":
        list_of_boxes, labels = [], []

        for it, bin_mask in enumerate(mask):
            if it == 0:
                continue

            boxes = mask2boxes(bin_mask.numpy)

            if boxes is None:
                continue

            list_of_boxes.extend(boxes)
            labels.extend([bin_mask.labels[-1]] * len(boxes))

        return BBoxListWrapper(list_of_boxes, labels)

    @property
    def numpy(self):
        return np.array([bbox.xyxy for bbox in self])

    @property
    def labels(self):
        return [bbox.label for bbox in self]

    def get(self, s: str):
        res = []

        for bbox in self:
            if bbox.label == s:
                res.append(bbox)

        return BBoxListWrapper(res)

    def cutout(self, image: ImageWrapper) -> list:
        npy = image.numpy
        return [
            ImageWrapper(npy[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]) for bbox in self
        ]
