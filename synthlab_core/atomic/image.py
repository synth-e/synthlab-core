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


class ImageWrapper(AtomicType):

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
        return cls(np.array(Image.open(BytesIO(b))))

    def to_buffer(self):
        buffer = BytesIO()
        h, w = self.size()
        
        if h > 2048 or w > 2048:
            # resize keep the aspect ratio
            if h > w:
                self.pil.resize((2048, int(2048 * h / w)), Image.LANCZOS).save(buffer, format="JPEG")
            else:
                self.pil.resize((int(2048 * w / h), 2048), Image.LANCZOS).save(buffer, format="JPEG")
        else:
            self.pil.save(buffer, format="JPEG")
        return buffer.getvalue()

    def to_web_compatible(self):
        return f"data:image/JPEG;base64,{base64.b64encode(self.to_buffer()).decode()}"

    def hash(self):
        data = self.to_web_compatible()
        return sha512(data.encode()).hexdigest()

    # return a np array
    def load_image(self, image):
        if isinstance(image, np.ndarray):
            return image.astype(np.uint8)

        if isinstance(image, IndexedFile):
            return self.load_image(image.path)

        if isinstance(image, ImageWrapper):
            return image.numpy.astype(np.uint8)

        if isinstance(image, torch.Tensor):
            return image.cpu().numpy().astype(np.uint8)

        if isinstance(image, Image.Image):
            return np.array(image).astype(np.uint8)

        if isinstance(image, str):
            buffer = None
            if image.startswith("http"):
                buffer = requests.get(image).content

            elif (
                image.startswith("data:image/png;base64,")
                or image.startswith("data:image/jpg;base64,")
                or image.startswith("data:image/jpeg;base64,")
            ):
                buffer = base64.b64decode(image.split(",")[1])

            elif os.path.exists(image):
                if image.endswith("png") or image.endswith("jpg"):
                    return np.array(Image.open(image)).astype(np.uint8)

                with open(image, "rb") as f:
                    buffer = f.read()

            if decltype.is_image(buffer):
                return np.array(Image.open(BytesIO(buffer))).astype(np.uint8)

            if decltype.is_nparray(buffer):
                return np.load(buffer).astype(np.uint8)

        raise ValueError(f"Unsupported image type: {type(image)}")

    @property
    def numpy(self):
        return self.image

    @property
    def tensor(self):
        return torch.from_numpy(self.numpy)

    @property
    def pil(self):
        return Image.fromarray(self.numpy)

    def __init__(self, image):
        self.image = self.load_image(image)
        
        if len(self.image.shape) == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        elif self.image.shape[-1] == 4:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGBA2RGB)
        elif self.image.shape[-1] == 1:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        # else:
        #     self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        assert (
            len(self.image.shape) == 3
        ), f"Image must be a 3D array, got {self.image.shape}"
        # assert self.image.shape[0] in [1, 3], f"Image must have 1 or 3 channels, got {self.image.shape[0]}"

        h, w = self.size()
        if h > 3192 or w > 3192:
            if h > w:
                self.image = cv2.resize(self.image, (3192, int(3192 * h / w)))
            else:
                self.image = cv2.resize(self.image, (int(3192 * w / h), 3192))

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, index):
        return self.image[index]

    def clone(self):
        return deepcopy(self)

    def size(self):
        return self.image.shape[:-1]

    def save(self, path):
        Image.fromarray(self.numpy).save(path)


class DiffusionResponse(ImageWrapper):

    def __init__(self, image, ca=None, sa=None, *args, **kwargs):
        super().__init__(image, *args, **kwargs)
        self._ca, self._sa = ca, sa

    @property
    def ca(self):
        return self._ca

    @property
    def sa(self):
        return self._sa
    
class Sketch(ImageWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)