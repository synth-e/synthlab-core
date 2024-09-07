from .base import AtomicType
from .image import ImageWrapper, Sketch, DiffusionResponse
from .mask import MaskWrapper
from .text import TextualPrompt, Identity
from .box import BBoxListWrapper, BBoxWrapper
from .index import IndexedFile

from synthlab_core.registry import register, ClassType

for e in [AtomicType, ImageWrapper, Sketch, 
     DiffusionResponse, MaskWrapper, 
     TextualPrompt, BBoxListWrapper, 
     BBoxWrapper, IndexedFile, Identity]: 
    register(ClassType.DATA_TYPE, e)