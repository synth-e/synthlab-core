from enum import Enum

__classes__ = {}


class ClassType(str, Enum):
    NODE = "node"
    MASTER_NODE = "master_node"
    DATA_TYPE = "data_type"


def register(cls_type: ClassType, cls) -> None:
    global __classes__
    
    print(f"Registering {cls_type}.{cls.__name__}")
    
    if cls_type == ClassType.NODE:
        inspecs = cls.in_specs()
        outspecs = cls.out_specs()
        
        if len(outspecs) == 0:
            raise ValueError(f"Output specs of {cls.__name__} is empty")

    if cls_type not in __classes__:
        __classes__[cls_type] = {}

    __classes__[cls_type][cls.__name__] = cls


def create(cls_type: ClassType, target, *args, **kwargs) -> object:
    global __classes__
    if cls_type not in __classes__:
        if kwargs.get("noexcept", False):
            return None

        raise ValueError(f"Class type {cls_type} not found")

    if target not in __classes__[cls_type]:
        if kwargs.get("noexcept", False):
            return None

        raise ValueError(f"Class {target} not found in {cls_type}")

    return __classes__[cls_type][target](*args, **kwargs)


def report() -> None:
    global __classes__
    for cls_type in __classes__:
        print(f"Class type {cls_type}:")
        for cls_name in __classes__[cls_type]:
            print(f"\t{cls_name}")


def describe(t: ClassType = ClassType.NODE) -> list:
    global __classes__

    res = []

    for name, cls in __classes__[t].items():
        res.append(cls.meta())

    return res


def print_instruction_of(class_name) -> None:
    for key, val in __classes__.items():
        if class_name in val:
            print(f"Instruction of {key}.{class_name}:")
            print(val[class_name].__doc__)


def all_classes_of(cls_type) -> list:
    if cls_type not in __classes__:
        return []

    return __classes__[cls_type].values()


def has_cls(cls_type: ClassType, target) -> bool:
    return cls_type in __classes__ and target in __classes__[cls_type]


def get_meta_of(target, cls_type: ClassType = ClassType.NODE) -> dict:
    if cls_type not in __classes__:
        return None

    if target not in __classes__[cls_type]:
        return None

    return __classes__[cls_type][target].meta()


def get_cls(cls_type: ClassType, target) -> type:
    if cls_type not in __classes__:
        return None

    if target not in __classes__[cls_type]:
        return None

    return __classes__[cls_type][target]


class ModuleNotFoundError(Exception):

    def __init__(self, cls_type: ClassType, target: str):
        self.cls_type = cls_type
        self.target = target

    def __str__(self):
        return (
            f"Class type {self.cls_type} not found"
            if self.target is None
            else f"Class {self.target} not found in {self.cls_type}"
        )
