import numpy as np




#-------------------------------------------------------------------#
# Abstract base memory class
#-------------------------------------------------------------------#
class BaseMemory():
    def __init__(self, buffer_size) -> None:
        self.buffer_size = buffer_size
    def store(self, *args, **kwargs):
        pass
    def sample(self, *args, **kwargs):
        pass
    def clear(self, *args, **kwargs):
        pass
