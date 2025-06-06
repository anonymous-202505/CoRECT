import heapq
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import click
import numpy as np
import torch

from corect.compression_registry import COMPRESSION_METHODS, add_compressions
from corect.model_wrappers import AbstractModelWrapper
from corect.quantization.BinaryCompression import BinaryCompression
from corect.utils import cos_sim, evaluate_results, hamming_distance


def search(
    model: AbstractModelWrapper,
    corpora: Dict[str, Dict[str, Dict[str, str]]],
    queries: Dict[str, str],
    top_k: int,
    chunk_size: int,
    dimensionalities: List[int],
    save_path: str = None,
    encode_kwargs: Dict[str, Any] = {},
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Search for the top-k documents for each query in the corpus.
    """
    # Initialize results dict
    add_compressions()
    query_ids = list(queries.keys())
    corpus_sizes = sorted(corpora.keys())
    results = {}
    result_heaps = defaultdict(dict)
    dimensionalities = sorted(dimensionalities, reverse=True)
    for dimensionality in dimensionalities:
        for quantization_name in COMPRESSION_METHODS.keys():
            for corpus_size in corpus_sizes:
                results[corpus_size] = results.get(corpus_size, defaultdict(dict))
                results[corpus_size][dimensionality][quantization_name] = {
                    qid: {} for qid in query_ids
                }

            # Initialize one heaps dict for all corpus sizes
            result_heaps[dimensionality][quantization_name] = {
                qid: [] for qid in query_ids
            }  # Keep only the top-k docs for each query

    # Embed queries or load saved embeddings:
    loaded_queries = False
    if save_path:
        try:
            query_embeddings = np.load(os.path.join(save_path, "queries.npy"))
            click.echo(f"Loaded query embeddings at {save_path}")
            loaded_queries = True
        except OSError:
            click.echo(f"Could not find any query embeddings at {save_path}.")

    if not loaded_queries:
        queries = [queries[qid] for qid in queries]
        query_embeddings = model.encode_queries(
            queries,
            **encode_kwargs,
        )

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "queries.npy"), query_embeddings)

    # Loop over corpora
    for corpus_size in corpus_sizes:
        click.echo(f"Encoding corpus of size {corpus_size}...")

        # Embed corpus
        corpus = corpora[corpus_size]
        corpus_ids = list(corpus.keys())
        corpus_ids = sorted(corpus_ids)
        corpus = [corpus[cid] for cid in corpus_ids]

        # Encoding corpus in batches... Warning: This might take a while!
        iterator = range(0, len(corpus), chunk_size)

        for batch_num, corpus_start_idx in enumerate(iterator):
            loaded_corpus = False
            if save_path:
                try:
                    sub_corpus_embeddings = np.load(
                        os.path.join(
                            save_path, str(corpus_size), f"corpus_batch_{batch_num}.npy"
                        )
                    )
                    click.echo(
                        f"Loaded corpus embeddings for batch {batch_num+ 1}/{len(iterator)} at {save_path}"
                    )
                    loaded_corpus = True
                except OSError:
                    click.echo(
                        f"Could not find any corpus embeddings for batch {batch_num + 1} at {save_path}."
                    )

            if not loaded_corpus:
                click.echo(f"Encoding Batch {batch_num + 1}/{len(iterator)}...")
                corpus_end_idx = min(corpus_start_idx + chunk_size, len(corpus))

                # Encode chunk of corpus
                sub_corpus_embeddings = model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    **encode_kwargs,
                )

                if save_path:
                    os.makedirs(
                        os.path.join(save_path, str(corpus_size)), exist_ok=True
                    )
                    np.save(
                        os.path.join(
                            save_path, str(corpus_size), f"corpus_batch_{batch_num}.npy"
                        ),
                        sub_corpus_embeddings,
                    )

            # Loop over all quantizations and dimensions
            for quantization_name, quantization in COMPRESSION_METHODS.items():
                query_embeds, corpus_embeds = None, None
                for dimensionality in dimensionalities:
                    # Quantize the embeddings
                    if query_embeds is None or corpus_embeds is None:
                        query_embeds = quantization.compress(
                            query_embeddings[:, :dimensionality]
                        )
                        corpus_embeds = quantization.compress(
                            sub_corpus_embeddings[:, :dimensionality]
                        )

                    _query_embeddings = query_embeds[:, :dimensionality].copy()
                    _sub_corpus_embeddings = corpus_embeds[:, :dimensionality].copy()

                    # Use cosine similarity for quantized embeddings, hamming distance for binarized
                    if not isinstance(quantization, BinaryCompression):
                        similarity_scores = cos_sim(
                            _query_embeddings, _sub_corpus_embeddings
                        )
                    else:
                        similarity_scores = hamming_distance(
                            _query_embeddings, _sub_corpus_embeddings
                        )

                    # Check for NaN values
                    assert torch.isnan(similarity_scores).sum() == 0

                    # Get top-k values
                    similarity_scores_top_k_values, similarity_scores_top_k_idx = (
                        torch.topk(
                            similarity_scores,
                            min(
                                top_k + 1,
                                (
                                    len(similarity_scores[1])
                                    if len(similarity_scores) > 1
                                    else len(similarity_scores[-1])
                                ),
                            ),
                            dim=1,
                            largest=True,
                        )
                    )
                    similarity_scores_top_k_values = (
                        similarity_scores_top_k_values.cpu().tolist()
                    )
                    similarity_scores_top_k_idx = (
                        similarity_scores_top_k_idx.cpu().tolist()
                    )

                    for query_itr in range(len(query_embeddings)):
                        query_id = query_ids[query_itr]
                        for sub_corpus_id, score in zip(
                            similarity_scores_top_k_idx[query_itr],
                            similarity_scores_top_k_values[query_itr],
                        ):
                            corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                            if (
                                len(
                                    result_heaps[dimensionality][quantization_name][
                                        query_id
                                    ]
                                )
                                < top_k
                            ):
                                # Push item on the heap
                                heapq.heappush(
                                    result_heaps[dimensionality][quantization_name][
                                        query_id
                                    ],
                                    (score, corpus_id),
                                )
                            else:
                                # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                                heapq.heappushpop(
                                    result_heaps[dimensionality][quantization_name][
                                        query_id
                                    ],
                                    (score, corpus_id),
                                )

        for dimensionality in dimensionalities:
            for quantization_name in COMPRESSION_METHODS.keys():
                for qid in result_heaps[dimensionality][quantization_name]:
                    for score, corpus_id in result_heaps[dimensionality][
                        quantization_name
                    ][qid]:
                        results[corpus_size][dimensionality][quantization_name][qid][
                            corpus_id
                        ] = score

    return results


def rc(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float]]:
    """
    Compute the Relevance Composition (RC) score for each query.

    Relevance composition @ k yields the proportion of retrieved documents that are
    relevant, distractors, and randoms among the top-ranked results.
    """
    RC = {}

    k_max, top_hits = max(k_values), {}
    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    query_relevant_docs = defaultdict(set)
    query_distractor_docs = defaultdict(set)
    for query_id in qrels:
        for doc_id in qrels[query_id]:
            if qrels[query_id][doc_id] == "relevant":
                query_relevant_docs[query_id].add(doc_id)
            elif qrels[query_id][doc_id] == "distractor":
                query_distractor_docs[query_id].add(doc_id)

    for k in k_values:
        relevant_count = 0
        distractor_count = 0
        for query_id in top_hits:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs[query_id]:
                    relevant_count += 1
                elif hit[0] in query_distractor_docs[query_id]:
                    distractor_count += 1
        RC[f"RC@{k}"] = {
            "relevant": round(relevant_count / len(top_hits), 5),
            "distractor": round(distractor_count / len(top_hits), 5),
        }

    return RC


def eval_results(
    dimensionalities: List[int],
    k_values: List[int],
    corpus_size: int | str,
    save_path: str,
    qrels_relevant_only: defaultdict,
    qrels: Optional[defaultdict],
    results: dict,
):
    for dimensionality in dimensionalities:
        for quantization in COMPRESSION_METHODS.keys():
            click.echo(
                f"Evaluating corpus {corpus_size}, dimensionality {dimensionality} and quantization {quantization}"
            )

            # Evaluate the results
            ndcg, _map, recall, precision, mrr = evaluate_results(
                qrels_relevant_only.copy(),
                results[corpus_size][dimensionality][quantization],
                k_values,
            )

            scores = {
                **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
                **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
                **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
                **{
                    f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()
                },
                **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            }

            # Custom evaluation
            if qrels:
                _rc = rc(
                    qrels,
                    results[corpus_size][dimensionality][quantization],
                    k_values,
                )
                for k, v in _rc.items():
                    scores[f"rc_at_{k.split('@')[1]}"] = v

            click.echo(f"NDCG@10: {scores['ndcg_at_10']}")

            # Save the results
            results_path = os.path.join(
                save_path,
                f"dim={dimensionality}",
                f"q={quantization}",
                f"{corpus_size}.json",
            )
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                json.dump(scores, f, indent=4)
