import numpy as np

from corect.quantization.AbstractCompression import AbstractCompression

PRECISION_TYPE = {
    "full": 32,
    "half": 32,
}


class FloatCompression(AbstractCompression):
    """
    Class for casting embedding vectors to float16 or leave them as full-precision vectors.
    """

    def __init__(self, precision_type: str = "half"):
        """
        Initializes the class with the precision type to use, i.e. full or half.

        Args:
            precision_type: The precision type.
        """
        assert precision_type in PRECISION_TYPE
        self.precision_type = precision_type

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Converts embeddings to float16, if the precision type is half, otherwise returns the full-precision vectors.

        Args:
            embeddings: The embeddings to convert.

        Returns:
            The converted embeddings.
        """
        if self.precision_type == "full":
            return embeddings
        elif self.precision_type == "half":
            return embeddings.astype(np.float16)
        else:
            raise NotImplementedError(
                f"Cannot convert embedding to invalid precision type {self.precision_type}!"
            )
