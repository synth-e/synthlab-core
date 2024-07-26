import datetime
import random
from inspect import signature
from synthlab_core.utilities.misc.decltype import type2readble
from hashlib import sha512

class INode(object):
    """Base class for all nodes in pipeline

    Notes
    -----
    INode must have inputs, outputs could exists or be None by default
    """

    def get_hashable(self):
        data = []

        hashable_data_type = (str, int, float, bool, type(None))
        ignore = ["inference_device", "idle_device", "low_resource_mode"]

        for k, v in self.__dict__.items():
            if k in ignore:
                continue

            if isinstance(v, hashable_data_type):
                data.append((k, v))
            else:
                data.append((k, str(v)))

        return data

    def __init__(self, *args, **kwargs) -> None:
        self._initialized = True
        self.inference_device = kwargs.get("inference_device", "cpu")
        self.low_resource_mode = kwargs.get("low_resource_mode", False)
        self.idle_device = "cpu"

    def set_inference_device(self, device):
        self.inference_device = device

    @property
    def initialized(self):
        return getattr(self, "_initialized", False)

    def hash(self):
        if not self.initialized:
            raise ValueError(
                f"Cannot hash an uninitialized object of {self.__class__.__name__}"
            )

        hashable = self.get_hashable()
        _sha512 = sha512()

        for k, v in hashable:
            _sha512.update(str(k).encode())
            _sha512.update(str(v).encode())

        _sha512.update(self.__class__.__name__.encode())

        return _sha512.hexdigest()

    # deprecated
    def generate_identity(self, *args, **kwargs):
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S") + str(
            random.randint(0, 9999)
        ).zfill(5)

    @classmethod
    def meta(cls):
        return {
            "name": cls.__name__,
            "in_specs": [[k, type2readble(v)] for k, v in cls.in_specs()],
            "out_specs": [[k, type2readble(v)] for k, v in cls.out_specs()],
            "params_specs": cls.params_specs(),
            "is_master_node": False,
            "friendly_name": cls.__name__,
            "instruction": cls.__doc__,
        }

    @classmethod
    def in_specs(cls) -> list[tuple[str, type]]:
        raise NotImplementedError(
            f"property in_dtype has not been implemented for {cls.__name__}"
        )

    @classmethod
    def out_specs(cls) -> list[tuple[str, type]]:
        return []

    @classmethod
    def params_specs(cls) -> list[tuple[str, type]]:
        return []

    @classmethod
    def sanity_format_check(cls):
        if not isinstance(cls.in_specs(), list):
            raise ValueError(
                f"Error in {cls.__name__}, input specs must be a list of tuple"
            )
        if not isinstance(cls.out_specs(), list):
            raise ValueError(
                f"Error in {cls.__name__}, output specs must be a list of tuple"
            )

        if len(cls.out_specs()) > 1:
            raise ValueError(
                f"Error in {cls.__name__}, length of output specs must 0 (nothing returns) or 1"
            )

        for spec in cls.in_specs() + cls.out_specs():
            if not isinstance(spec, tuple):
                raise ValueError(
                    f"Error in {cls.__name__}, each item in specs must be a tuple"
                )

            if len(spec) != 2:
                raise ValueError(
                    f"Error in {cls.__name__}, each tuple item in specs must have 2 items [name_in_config: str, TypeName: AtomicType]"
                )

            if not isinstance(spec[0], str):
                raise ValueError(
                    f"Error in {cls.__name__}, first item of tuple spec must be a string, specifing name_in_config"
                )

            # TODO: check if input and output are atomic types

        sig = signature(cls.__call__)
        call_arg_count = 0  # (not count args and kwargs)
        # TODO: better way to check if arg is args-liked parameter
        for arg, type in sig.parameters.items():
            call_arg_count += "args" not in arg and arg != "self"

        if call_arg_count != len(cls.in_specs()):
            raise ValueError(
                f"Input specs of module {cls.__name__} do not match with __call__ args, input specs count={len(cls.in_specs())}, __call__ args count={call_arg_count}"
            )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"__call__ has not been implemented for {self.__class__.__name__}"
        )


class IMasterNode(object):
    """Base class for master nodes in pipeline.

    Notes
    -----
    Master nodes do not have inputs, only outputs
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def generate_identity(self, *args, **kwargs):
        return datetime.datetime.now().strftime("%Y%m%d%H%M%S") + str(
            random.randint(0, 9999)
        ).zfill(5)

    @classmethod
    def meta(cls):
        return {
            "name": cls.__name__,
            "in_specs": [[k, type2readble(v)] for k, v in cls.in_specs()],
            "out_specs": [[k, type2readble(v)] for k, v in cls.out_specs()],
            "params_specs": cls.params_specs(),
            "is_master_node": True,
            "friendly_name": cls.__name__,
            "instruction": cls.__doc__,
        }

    @classmethod
    def out_specs(cls) -> list[tuple[str, type]]:
        raise NotImplementedError(
            f"property out_dtype has not been implemented for {cls.__name__}"
        )

    @classmethod
    def in_specs(cls) -> list[tuple[str, type]]:
        return []

    @classmethod
    def params_specs(cls) -> list[tuple[str, type]]:
        return []

    @classmethod
    def sanity_format_check(cls):
        if len(cls.out_specs()) != 1:
            raise ValueError(
                f"Error in {cls.__name__}, length of output specs must be equal to 1"
            )

        for spec in cls.out_specs():
            if not isinstance(spec, tuple):
                raise ValueError(
                    f"Error in {cls.__name__}, each item in specs must be a tuple"
                )

            if len(spec) != 2:
                raise ValueError(
                    f"Error in {cls.__name__}, each tuple item in specs must have 2 items [name_in_config: str, TypeName: AtomicType]"
                )

            if not isinstance(spec[0], str):
                raise ValueError(
                    f"Error in {cls.__name__}, first item of tuple spec must be a string, specifing name_in_config"
                )

            # TODO: check if input and output are atomic types

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"__call__ has not been implemented for {self.__class__.__name__}"
        )
