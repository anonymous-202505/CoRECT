from abc import ABC, abstractmethod
from typing import *

import numpy as np
import torch


class AbstractModelWrapper(ABC):

    @abstractmethod
    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        Encode a list of queries into embeddings.
        """
        pass

    @abstractmethod
    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        """
        Encode a list of documents into embeddings.
        """
        pass

    @abstractmethod
    def similarity(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the similarity between two batches of embeddings.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the name of the model.
        """
        pass
