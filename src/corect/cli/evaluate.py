import os
import time
from typing import Dict, List

import click

from corect.config import *
from corect.dataset_utils import DATASETS, load_data
from corect.eval_utils import eval_results, search
from corect.model_wrappers import (
    AbstractModelWrapper,
    E5MultilingualWrapper,
    JinaV3Wrapper,
)


def _get_model_wrapper(model_name: str) -> AbstractModelWrapper:
    """
    Attempts to return the model wrapper used to create embeddings to be evaluated and quantized. If the given model is
    unsupported, an error is thrown instead.

    Args:
        model_name: The model to use.

    Returns:
        The model wrapper, if implemented.
    """
    if model_name == "jina":
        return JinaV3Wrapper()
    elif model_name == "e5":
        return E5MultilingualWrapper()
    else:
        raise NotImplementedError(f"Model {model_name} not supported!")


def _get_dataset(dataset_name: str) -> List[str] | Dict[str, Dict[str, int]]:
    """
    Returns the dataset dictionary or list containing the names of the datasets to load.

    Args:
        dataset_name: The name of the dataset.

    Returns:
        The list or dictionary containing the dataset names.
    """
    if dataset_name in DATASETS.keys():
        return DATASETS[dataset_name]
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not supported!")


@click.command()
@click.argument("model", type=str)
@click.argument("dataset", type=str)
def evaluate(model: str, dataset: str):
    """
    Evaluates the retrieval performance for the given dataset using different quantization methods.

    Args:
        model: A string representing the name of the embedding model (has to be supported by _get_model_wrapper()).
        dataset: A string representing the name of the dataset (has to be supported by _get_dataset()).
    """
    embed_model = _get_model_wrapper(model)
    embed_data = _get_dataset(dataset)
    click.echo(f"Loaded model: {embed_model.name}")

    for data in embed_data:
        click.echo(f"Starting evaluation on dataset {dataset}:{data}")
        embed_folder = os.path.join(EMBED_FOLDER, model, data)
        corpora, queries, qrels, qrels_relevant_only = load_data(dataset, data)

        # Evaluate the dataset
        start = time.time()
        results = search(
            embed_model,
            corpora,
            queries,
            max(K_VALUES),
            CORPUS_CHUNK_SIZE,
            DIMENSIONALITIES,
            embed_folder,
        )
        end = time.time()
        click.echo(f"Time taken: {end - start:.2f} seconds")

        # Loop over corpora
        for corpus_size in sorted(corpora.keys()):
            save_path = os.path.join(RESULTS_FOLDER, model, data)
            eval_results(
                DIMENSIONALITIES,
                K_VALUES,
                corpus_size,
                save_path,
                qrels_relevant_only,
                qrels,
                results,
            )
