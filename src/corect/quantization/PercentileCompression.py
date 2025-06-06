import numpy as np

from corect.quantization.AbstractCompression import AbstractCompression

NUM_BITS = [2, 4, 8]


class PercentileCompression(AbstractCompression):
    """
    Class for quantizing embedding vectors using bins containing the same number of points.
    """

    def __init__(self, num_bits: int = 8):
        """
        Initializes the class with the number of bits the embeddings should be quantized to.

        Args:
            num_bits: The number of bits the resulting embeddings should occupy.
        """
        assert 1 < num_bits < 16
        self.num_bits = num_bits

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantizes the embeddings by calculating all their 2**num_bits percentiles per embedding dimension. The
        percentile values are then used as boundaries such that embedding values are converted to the respective bin
        number.

        Args:
            embeddings: The embeddings to quantize.

        Returns:
            The quantized embeddings.
        """
        quantiles = np.linspace(0, 100, num=2**self.num_bits + 1)
        bin_edges = np.percentile(embeddings, quantiles, axis=0)
        bin_indices = np.empty_like(embeddings, dtype=int)
        for col in range(embeddings.shape[1]):
            # Getting the corresponding bin indices from the values
            bin_indices[:, col] = np.digitize(
                embeddings[:, col], bin_edges[1:-1, col], right=False
            )
        return bin_indices
