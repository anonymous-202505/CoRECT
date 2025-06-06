from abc import ABC, abstractmethod

import numpy as np


class AbstractCompression(ABC):
    """
    Base class for compression methods.
    """

    @abstractmethod
    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compress the embeddings and return them.
        """
        pass
