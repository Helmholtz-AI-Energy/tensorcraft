from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from tensorcraft.tensor import Tensor
from tensorcraft.types import MIndex


class Dist(ABC):
    @abstractmethod
    def processorView(self, tensor: Tensor) -> np.ndarray:
        pass

    # @abstractmethod
    # def info(self) -> None:
    #     pass

    @property
    @abstractmethod
    def numProcessors(self) -> int:
        pass

    @property
    @abstractmethod
    def processorArrangement(self) -> npt.NDArray[np.int_]:
        pass

    @abstractmethod
    def getProcessorMultiIndex(self, index: int) -> MIndex:
        pass

    @abstractmethod
    def getIndexLocation(self, tensor: Tensor, index: MIndex) -> MIndex:
        pass

    @abstractmethod
    def compatible(self, tensor: Tensor) -> bool:
        pass
