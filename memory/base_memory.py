import numpy as np
from abc import ABC




#-------------------------------------------------------------------#
# Abstract base memory class
#-------------------------------------------------------------------#
class BaseMemory(ABC):
    def __init__(self, buffer_size) -> None:
        super().__init__()
        self.buffer_size = buffer_size
    def store(self, *args, **kwargs):
        pass
    def sample(self, *args, **kwargs):
        pass
    def clear(self, *args, **kwargs):
        pass
