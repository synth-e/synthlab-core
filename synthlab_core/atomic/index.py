from .base import AtomicType

class IndexedFile(AtomicType):
    """
    Deprecated
    """

    def __init__(self, identity, path) -> None:
        self._identity = identity
        self._path = path

    @property
    def identity(self):
        return self._identity

    @property
    def path(self):
        return self._path

    def __repr__(self) -> str:
        return f"IndexedFile({self.identity}, {self.path})"

    def __str__(self) -> str:
        return self.__repr__()
