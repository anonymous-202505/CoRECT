from typing import Dict

from corect.quantization.AbstractCompression import AbstractCompression
from corect.quantization.BinaryCompression import THRESHOLD_TYPES, BinaryCompression
from corect.quantization.FloatCompression import PRECISION_TYPE, FloatCompression
from corect.quantization.MinMaxCompression import MinMaxCompression
from corect.quantization.PercentileCompression import NUM_BITS, PercentileCompression

# Dictionary containing all compression methods to be used by the evaluate script.
COMPRESSION_METHODS: Dict[str, AbstractCompression] = {}


def add_compressions():
    for precision, num_bits in PRECISION_TYPE.items():
        COMPRESSION_METHODS[num_bits] = FloatCompression(precision)
    for bits in NUM_BITS:
        COMPRESSION_METHODS[f"{bits}_percentile"] = PercentileCompression(bits)
        COMPRESSION_METHODS[f"{bits}_equal_distance"] = MinMaxCompression(bits, 2.5)
    for threshold in THRESHOLD_TYPES:
        COMPRESSION_METHODS[f"1_binary_{threshold}"] = BinaryCompression(threshold)
