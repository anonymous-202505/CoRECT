import numpy as np
import pytrec_eval
import torch


def count_lines(filepath):
    with open(filepath, "r") as f:
        return sum(1 for _ in f)


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).float()

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b).float()

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.

    Return:
        Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def hamming_distance(a: torch.Tensor, b: torch.Tensor):
    """
    Computes the hamming distance between two binary vectors.

    Return:
        Matrix with res[i][j]  = hamming_distance(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a).float()

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b).float()

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return -1 * torch.cdist(a, b, p=1)


def mrr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[dict[str, float]]:
    MRR = {}

    for k in k_values:
        MRR[f"MRR@{k}"] = []

    k_max, top_hits = max(k_values), {}

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(
            doc_scores.items(), key=lambda item: item[1], reverse=True
        )[0:k_max]

    for query_id in top_hits:
        query_relevant_docs = {
            doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0
        }
        for k in k_values:
            rr = 0
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    rr = 1.0 / (rank + 1)
                    break
            MRR[f"MRR@{k}"].append(rr)

    return MRR


def _quantize_embeddings(
    embeddings: np.ndarray, cutoffs: list, bit_nums: list
) -> np.ndarray:
    """
    Quantizes the given embeddings to uint8 using the minimum and maximum value per embedding dimension with a value
    range according to the number of bits specified in the respective list. The cutoffs list is used to specify parts of
    the embedding that should be quantized differently, i.e. cutoffs = [512, 1024] and bit_nums = [8, 4] says that the
    first 512 dimensions of each embedding vector should be quantized to a range of 8bits while the next 512 dimensions
    should be quantized to a 4bit range.

    :param embeddings: The embeddings to quantize.
    :param cutoffs: The parts of the embeddings that should be quantized to the same range.
    :param bit_nums: The number of bits indicating the range of the quantized embeddings per cutoff point.
    :return: The quantized embeddings.
    """
    embed_start = 0
    bin_vector = None
    mins = np.min(embeddings, axis=0)
    maxs = np.max(embeddings, axis=0)
    for dim, cutoff in enumerate(cutoffs):
        bit_num = bit_nums[dim]
        sub_embed = embeddings[:, embed_start:cutoff]
        ranges = np.vstack((mins[embed_start:cutoff], maxs[embed_start:cutoff]))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / (2**bit_num - 1)
        sub_embed = ((sub_embed - starts) / steps).numpy().astype(np.uint8)

        if bin_vector is None:
            bin_vector = sub_embed
        else:
            bin_vector = np.concatenate((bin_vector, sub_embed), axis=1)
        embed_start = cutoff
    return bin_vector


def _validate_cutoffs(cutoffs: list, bit_nums: list, max_len: int):
    """
    Validates that the cutoffs and bit_nums lists are of the same length and contain valid numbers.

    :param cutoffs: The cutoff points in the embedding vector.
    :param bit_nums: The number of bits to use per cutoff point.
    """
    assert len(cutoffs) == len(bit_nums)
    last_dim = 0
    for idx, cutoff in enumerate(cutoffs):
        if cutoff <= last_dim:
            raise ValueError(
                f"Illegal cutoff point {cutoff} smaller than minimum allowed dimension {last_dim}!"
            )
        elif cutoff > max_len:
            raise ValueError(
                f"Illegal cutoff point {cutoff} bigger than the maximum allowed dimension {max_len}!"
            )
        elif bit_nums[idx] < 2 or bit_nums[idx] > 8:
            raise ValueError(
                f"Number of bits has to be between 2 and 8 but was {bit_nums[idx]}"
            )


def min_max_quantization(
    embeddings: np.ndarray, cutoffs: list, bit_nums: list
) -> np.ndarray:
    """
    Quantizes embeddings using the minimum and maximum per embedding dimension. The range of the resulting embeddings
    is determined by the number of bits. The cutoff points are used to quantize different parts of the embeddings to
    a different ranges, i.e. cutoffs = [512, 1024] and bit_nums = [8, 4] will quantize the first 512 dimensions to a
    range of 8bits and the next 512 dimensions to a 4bit range.

    :param embeddings: The embeddings to quantize.
    :param cutoffs: The parts of the embeddings that should be quantized to the same range.
    :param bit_nums: The number of bits to use per cutoff point.
    :return: The quantized embeddings.
    """
    max_len = embeddings.shape[1]
    if len(cutoffs) == 0:
        cutoffs = [max_len]
    _validate_cutoffs(cutoffs, bit_nums, max_len)
    return _quantize_embeddings(embeddings, cutoffs, bit_nums)


def threshold_binarization(embeddings: np.ndarray, threshold: float = 0) -> np.ndarray:
    """
    Returns a binary representation of the embeddings based on the given threshold. Everything above the threshold is
    converted to 1 while everything equal to and below is 0.

    :param embeddings: The embeddings to binarize.
    :param threshold: The threshold to use.
    :return: The binarized embeddings.
    """
    return np.packbits(embeddings > threshold).reshape(embeddings.shape[0], -1)


def evaluate_results(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: list[int],
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    all_ndcgs, all_aps, all_recalls, all_precisions = {}, {}, {}, {}

    for k in k_values:
        all_ndcgs[f"NDCG@{k}"] = []
        all_aps[f"MAP@{k}"] = []
        all_recalls[f"Recall@{k}"] = []
        all_precisions[f"P@{k}"] = []

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string}
    )
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            all_ndcgs[f"NDCG@{k}"].append(scores[query_id]["ndcg_cut_" + str(k)])
            all_aps[f"MAP@{k}"].append(scores[query_id]["map_cut_" + str(k)])
            all_recalls[f"Recall@{k}"].append(scores[query_id]["recall_" + str(k)])
            all_precisions[f"P@{k}"].append(scores[query_id]["P_" + str(k)])

    ndcg, _map, recall, precision = (
        all_ndcgs.copy(),
        all_aps.copy(),
        all_recalls.copy(),
        all_precisions.copy(),
    )
    _mrr = mrr(qrels, results, k_values)

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(sum(ndcg[f"NDCG@{k}"]) / len(scores), 5)
        _map[f"MAP@{k}"] = round(sum(_map[f"MAP@{k}"]) / len(scores), 5)
        recall[f"Recall@{k}"] = round(sum(recall[f"Recall@{k}"]) / len(scores), 5)
        precision[f"P@{k}"] = round(sum(precision[f"P@{k}"]) / len(scores), 5)
        _mrr[f"MRR@{k}"] = round(sum(_mrr[f"MRR@{k}"]) / len(scores), 5)

    return ndcg, _map, recall, precision, _mrr
