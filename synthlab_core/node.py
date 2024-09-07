from inspect import signature
from synthlab_core.utilities.misc.decltype import type2readble
from hashlib import sha512
import torch

class INode(torch.nn.Module):
    """Base class for all nodes in pipeline

    Notes
    -----
    INode must have inputs, outputs could exists or be None by default
    """

    def get_hashable(self):
        return self.cache_keys

    def __init__(self, *args, **kwargs) -> None:
        super(INode, self).__init__()

        self.lazy_pool = {}
        self._initialized = True
        self.inference_device = kwargs.get("inference_device", "cpu")
        self.switch_device = kwargs.get("switch_device", False)
        self.idle_device = "cpu"
        self.cache_keys = kwargs.get("cache_keys", [])

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

    def ready(self, force=False):
        if self.switch_device or force:
            self.to(self.inference_device)

    def idle(self, force=False):
        if self.switch_device or force:
            self.to(self.idle_device)

    def lazy_update(self, key, val):
        if val != getattr(self, key):
            self.lazy_pool[key] = val

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
    def out_dtype(self):
        if len(self.out_specs()) == 0:
            return None
        
        return self.out_specs()[0][1]
    
    @classmethod
    def in_dtype(self, key):
        
        for k in self.in_specs():
            if k[0] == key:
                return k[1]
        
        return None

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

        sig = signature(cls.forward)
        call_arg_count = 0  # (not count args and kwargs)
        # TODO: better way to check if arg is args-liked parameter
        for arg, type in sig.parameters.items():
            call_arg_count += "args" not in arg and arg != "self"

        if call_arg_count != len(cls.in_specs()):
            raise ValueError(
                f"Input specs of module {cls.__name__} do not match with forward args, input specs count={len(cls.in_specs())}, __call__ args count={call_arg_count}"
            )

    def lazy_reload(self):
        prev = {}

        for k, v in self.lazy_pool.items():
            if hasattr(self, k) and getattr(self, k) != v:
                prev[k] = getattr(self, k)
                setattr(self, k, v)

        return prev

    def __call__(self, *args, **kwargs):        
        self.lazy_reload()

        try:
            self.ready()
            return super(INode, self).__call__(*args, **kwargs)
        except Exception as e:
            raise e
        finally:
            self.idle()

    def __batch_call__(self, inputs, **kwargs):        
        self.ready(True)
        for item in inputs:
            yield self.forward(*item, **kwargs)
        self.idle(True)


class IMasterNode(torch.nn.Module):
    """Base class for master nodes in pipeline.

    Notes
    -----
    Master nodes do not have inputs, only outputs
    """

    def __init__(self, *args, **kwargs) -> None:
        super(IMasterNode, self).__init__()

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

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"__call__ has not been implemented for {self.__class__.__name__}"
        )