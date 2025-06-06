import numpy as np

from corect.quantization.AbstractCompression import AbstractCompression

NUM_BITS = [2, 4, 8]


class MinMaxCompression(AbstractCompression):
    """
    Class for quantizing embedding vectors using the minimum and maximum values per embedding dimension.
    """

    def __init__(self, num_bits: int = 8, clip_percentile: float = 0):
        """
        Initializes the class with the number of bits and the percentiles to clip.

        Args:
            num_bits: The number of bits the resulting embeddings should occupy.
            clip_percentile: The percentile of outliers that should be clipped before quantization.
        """
        assert 1 < num_bits < 16
        assert 0 <= clip_percentile < 50
        self.num_bits = num_bits
        self.clip_percentile = clip_percentile

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Quantizes the embeddings by first clipping a fixed percentile of minimum and maximum values per embedding
        dimension (optional) before using the minimum and maximum value per dimension to calculate the bin boundaries
        using 2**num_bits - 1 steps of equal length from minimum to maximum. The embedding values are then converted to
        their respective bin numbers.

        Args:
            embeddings: The embeddings to quantize.

        Returns:
            The quantized embeddings.
        """
        out_embeds = embeddings
        if self.clip_percentile > 0:
            min_perc, max_perc = self.clip_percentile, 100 - self.clip_percentile
            points = np.percentile(embeddings, [min_perc, max_perc], axis=0)
            out_embeds = np.clip(embeddings, points[0], points[1])
        mins = np.min(out_embeds)
        maxs = np.max(out_embeds)
        steps = (maxs - mins) / (2**self.num_bits - 1)
        return np.floor((out_embeds - mins) / steps) - int(2**self.num_bits * 0.5)
