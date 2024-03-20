from abc import ABC, abstractmethod

import numpy as np

from tensorcraft.tensor import Tensor


class Dist(ABC):
    @abstractmethod
    def processorView(self, tensor: Tensor) -> np.ndarray:
        pass

    # @abstractmethod
    # def info(self) -> None:
    #     pass

    @property
    @abstractmethod
    def numProcessors(self) -> int | np.int64:
        pass

    @property
    @abstractmethod
    def processorArrangement(self) -> np.ndarray:
        pass

    @abstractmethod
    def getProcessorMultiIndex(self, index: int) -> tuple:
        pass

    @abstractmethod
    def getIndexLocation(
        self, tensor: Tensor, index: int | tuple | np.ndarray
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compatible(self, tensor: Tensor) -> bool:
        pass
