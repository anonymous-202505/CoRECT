import numpy as np

from corect.quantization.AbstractCompression import AbstractCompression

THRESHOLD_TYPES = ["zero", "median"]


class BinaryCompression(AbstractCompression):
    """
    Class for binary compression using a threshold to binarize embedding vectors.
    """

    def __init__(self, threshold_type: str = "zero"):
        """
        Initializes the class with the type of threshold to use.

        Args:
            threshold_type: The type of threshold to use, i.e. the median per embedding dimension or zero.
        """
        assert threshold_type in THRESHOLD_TYPES
        self.threshold_type = threshold_type

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Binarizes the given embeddings vectors using a zero threshold or the median per embedding dimension, depending
        on the defined threshold type.

        Args:
            embeddings: The embeddings to compress.

        Returns:
            The binarized embedding vectors.
        """
        if self.threshold_type == "zero":
            threshold = 0
        else:
            threshold = np.median(embeddings, axis=0)

        return np.where(embeddings > threshold, 1, 0)
