
class AtomicType(object):

    @classmethod
    def from_file(cls, p):
        raise NotImplementedError(
            f"from_file is not implemented for class {cls.__name__}"
        )

    @classmethod
    def from_buffer(cls, b: bytes):
        raise NotImplementedError(
            f"from_buffer is not implemented for class {cls.__name__}"
        )

    def to_file(self, p: str):
        raise NotImplementedError("to_file is not implemented")

    def to_buffer(self) -> bytes:
        raise NotImplementedError("to_bytes is not implemented")

    def to_web_compatible(self):
        return self.to_buffer()

    def hash(self):
        raise NotImplementedError(
            f"hash method is not implemented for class {self.__class__.__name__}"
        )

    def blend(self, board):
        raise NotImplementedError(
            f"blend method is not implemented for class {self.__class__.__name__}"
        )