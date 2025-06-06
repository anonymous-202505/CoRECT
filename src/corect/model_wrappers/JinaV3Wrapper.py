from typing import *

from transformers import AutoModel

from corect.model_wrappers import AbstractModelWrapper
from corect.utils import *


def _construct_document(doc):
    if isinstance(doc, str):
        return doc
    elif "title" in doc:
        return f"{doc['title']} {doc['text'].strip()}"
    else:
        return doc["text"].strip()


class JinaV3Wrapper(AbstractModelWrapper):
    def __init__(
        self,
        pretrained_model_name="jinaai/jina-embeddings-v3",
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name, trust_remote_code=True
        )
        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self.encoder.encode(sentences, *args, task="retrieval.query", **kwargs)

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        _sentences = [_construct_document(sentence) for sentence in sentences]
        return self.encoder.encode(
            _sentences, *args, task="retrieval.passage", **kwargs
        )

    def get_instructions(self):
        return [
            self.encoder._task_instructions[x]
            for x in ["retrieval.query", "retrieval.passage"]
        ]

    def similarity(
        self, embeddings_1: torch.Tensor, embeddings_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the similarity between two batches of embeddings.
        """
        return cos_sim(embeddings_1, embeddings_2)

    @property
    def device(self):
        return self.encoder.device

    @staticmethod
    def has_instructions():
        return True

    @property
    def name(self) -> str:
        """
        Return the name of the model.
        """
        return "JinaV3Wrapper"
