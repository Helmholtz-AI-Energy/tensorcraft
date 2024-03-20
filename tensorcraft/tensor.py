import numpy as np

from tensorcraft.util import multi2linearIndex, order2npOrder


class Tensor:
    def __init__(self, dims: tuple | int) -> None:
        if isinstance(dims, int):
            self._dims = np.array((dims,), dtype=np.int32)
        elif len(dims) < 1:
            raise ValueError("Tensor must have at least one dimension")
        elif len(dims) == 1:
            self._dims = np.array((dims[0],), dtype=np.int32)
        else:
            self._dims = np.array(dims, dtype=np.int32)

    @property
    def order(self) -> int:
        return len(self._dims)

    @property
    def shape(self) -> np.ndarray:
        return self._dims

    @property
    def size(self) -> np.int64:
        return np.prod(self._dims)

    def linearIndex(self, indices: tuple, order: str | tuple = "R") -> int:
        if len(indices) != len(self._dims):
            raise ValueError(
                "Indices must have the same length as the tensor's dimensions"
            )
        idx_array = np.array(indices)

        if order == "R":
            # print(np.arange(len(self._dims))[::-1])
            return multi2linearIndex(
                self._dims, idx_array, order=np.arange(len(self._dims))[::-1]
            )
        elif order == "C":
            return multi2linearIndex(self._dims, idx_array)
        elif isinstance(order, (list, tuple)):
            order_array = np.array(order)
            return multi2linearIndex(self._dims, idx_array, order_array)
        else:
            raise ValueError("Invalid order")

    def getMultiIndex(self, index: int, order: str = "R") -> tuple:
        if index < 0 or index >= self.size:
            raise ValueError("Index out of bounds")

        return np.unravel_index(index, self._dims, order=order2npOrder(order))

    def info(self) -> None:
        print(f"Order: {self.order}")
        print(f"Shape: {self.shape}")
        print(f"Size: {self.size}")
