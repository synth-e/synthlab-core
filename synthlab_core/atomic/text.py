from .base import AtomicType
import json
import numpy as np
import nltk
from hashlib import sha512


class TextualPrompt(AtomicType):

    @classmethod
    def from_file(cls, p: str) -> "TextualPrompt":
        if p.endswith(".json"):
            with open(p, "r") as f:
                return cls(**json.load(f))

        if p.endswith(".txt"):
            with open(p, "r") as f:
                return cls(text=f.read())

        raise ValueError(f"Unsupported file type: {p}")

    @classmethod
    def from_buffer(cls, b: bytes) -> "TextualPrompt":
        return cls(**json.loads(b.decode()))

    def to_buffer(self) -> bytes:
        return json.dumps(self.json).encode()

    def to_web_compatible(self):
        return self.text

    def hash(self):
        data = self.to_web_compatible()

        sha = sha512(data.encode())

        # if len(self._text) > 0: # specific for sd # not stable
        #     sha.update(self.seed.to_bytes(4, byteorder="big"))

        return sha.hexdigest()

    def __init__(
        self, text=None, labels=None, properties=None, seed=None, *args, **kwargs
    ):
        self._text = text or ""
        self._labels = labels or []
        self._properties = properties or {}
        self._seed = seed or np.random.randint(0, 1000000)

    @property
    def text(self):
        if len(self._text) > 0:
            return self._text

        if len(self.labels) > 0:
            return ", ".join(self._labels)

        if len(self._properties) > 0:
            return ", ".join(
                value + " " + key for key, value in self._properties.items()
            )

        return ""

    @property
    def labels(self):
        if len(self._labels) > 0:
            return self._labels

        if len(self._properties) > 0:
            return [value + " " + key for key, value in self._properties.items()]

        return [e.strip() for e in self._text.split(",")]

    @property
    def properties(self):
        return self._properties

    @property
    def seed(self):
        return self._seed

    @property
    def nouns(self):
        tokenized = nltk.word_tokenize(self.text)
        nouns = [
            (i, word)
            for i, (word, pos) in enumerate(nltk.pos_tag(tokenized))
            if pos[:2] == "NN"
        ]

        return nouns

    @property
    def json(self):
        return {
            "text": self.text,
            "labels": self.labels,
            "properties": self.properties,
            "seed": self.seed,
        }

    def __str__(self) -> str:
        return f"Prompt(text='{self.text}', labels={self.labels}, properties={self.properties}, seed={self.seed})"



