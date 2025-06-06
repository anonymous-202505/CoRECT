import json
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict

from corect.config import *

USE_QUANTIZATION_METHODS = [
    "16_casting",
    "8_percentile",
    "4_percentile",
    "2_percentile",
    "1_median",
]
METRICS = {
    "ndcg_at_10": "NDCG@10",
    "recall_at_100": "Recall@100",
    "recall_at_1000": "Recall@1000",
}
DATASETS = {
    "passage": [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
        100_000_000,
    ],
    "document": [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    ],
    "arguana": ["arguana"],
    # "climate-fever": ["climate-fever"],
    "cqadupstack-android": ["cqadupstack-android"],
    "cqadupstack-english": ["cqadupstack-english"],
    "cqadupstack-gaming": ["cqadupstack-gaming"],
    "cqadupstack-gis": ["cqadupstack-gis"],
    "cqadupstack-mathematica": ["cqadupstack-mathematica"],
    "cqadupstack-physics": ["cqadupstack-physics"],
    "cqadupstack-programmers": ["cqadupstack-programmers"],
    "cqadupstack-stats": ["cqadupstack-stats"],
    "cqadupstack-tex": ["cqadupstack-tex"],
    "cqadupstack-unix": ["cqadupstack-unix"],
    "cqadupstack-webmasters": ["cqadupstack-webmasters"],
    "cqadupstack-wordpress": ["cqadupstack-wordpress"],
    "dbpedia": ["dbpedia"],
    # "fever": ["fever"],
    "fiqa": ["fiqa"],
    # "hotpotqa": ["hotpotqa"],
    "nfcorpus": ["nfcorpus"],
    "nq": ["nq"],
    "quora": ["quora"],
    "scidocs": ["scidocs"],
    "scifact": ["scifact"],
    "touche2020": ["touche2020"],
    "trec-covid": ["trec-covid"],
}


def collect_data(model_name: str, document_length: str, corpus_size: int, metric: str):
    results_path = os.path.join(
        SHARE_RESULTS_FOLDER,
        model_name,
        document_length,
    )

    heatmap_data = defaultdict(dict)
    for dim_dir in os.listdir(results_path):
        dim_path = os.path.join(results_path, dim_dir)
        if not os.path.isdir(dim_path) or not dim_dir.startswith("dim="):
            continue

        dim = int(dim_dir.split("=")[1])

        for q_dir in os.listdir(dim_path):
            q_path = os.path.join(dim_path, q_dir)
            if not os.path.isdir(q_path) or not q_dir.startswith("q="):
                continue

            q_method = q_dir.split("=")[1]
            if q_method not in USE_QUANTIZATION_METHODS:
                continue
            q = int(q_method.split("_")[0])

            json_file = os.path.join(q_path, f"{corpus_size}.json")

            if os.path.isfile(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                    m = data.get(metric)
                    if m is not None:
                        heatmap_data[q][dim] = m * 100

    return heatmap_data


def plot_heatmap(
    data: dict, model_name: str, document_length: str, corpus_size: int, metric: str
):
    plot_path = os.path.join(
        RESOURCES_FOLDER,
        "heatmaps",
        model_name,
        metric,
        f"{document_length}_{corpus_size}.pdf",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    df = (
        pd.DataFrame(data)
        .T.sort_index(ascending=False)
        .sort_index(axis=1, ascending=False)
    )
    # print(df)
    # import sys; sys.exit(0)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 14},
        cmap="crest_r",
    )

    # Get the colorbar and make it larger
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(f"{METRICS[metric]} in %", fontsize=10)

    plt.xlabel("Dimensionality", fontsize=14, labelpad=10)
    plt.ylabel("Number of Bits per Dimension", fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace(".pdf", ".png"), dpi=300)
    plt.close()


def plot_aggregated_heatmap(model_name: str, metric: str, datasets: dict):
    arrays = []
    cqa_values = []
    cqa_scores = {}
    save_path = os.path.join(
        RESOURCES_FOLDER, "heatmaps", model_name, metric, "aggregated.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for document_length, corpus_sizes in datasets.items():

        for corpus_size in corpus_sizes:
            if isinstance(corpus_size, int):
                continue

            # Collect data from the model directory
            data = collect_data(model_name, document_length, corpus_size, metric)
            full_prec_value = data[16][1024]
            data = {
                key: dict(sorted(value.items(), key=lambda x: x[0], reverse=True))
                for key, value in data.items()
            }
            data = dict(sorted(data.items(), key=lambda x: x[0], reverse=True))
            values = []
            for dim, scores in data.items():
                for q, score in scores.items():
                    values.append(score - full_prec_value)
            if "cqadupstack" in corpus_size:
                cqa_values.append(np.array(values))
                for dim, scores in data.items():
                    if dim not in cqa_scores.keys():
                        cqa_scores[dim] = {}
                    for q, score in scores.items():
                        if q in cqa_scores[dim].keys():
                            cqa_scores[dim][q].append(score)
                        else:
                            cqa_scores[dim][q] = [score]
            else:
                arrays.append(np.array(values))

    # Skip if no data is available
    if not arrays and not cqa_values:
        click.echo("No data available for the heatmap.")
        return

    for dim, scores in cqa_scores.items():
        for q, score in scores.items():
            cqa_scores[dim][q] = np.mean(cqa_scores[dim][q])

    plot_heatmap(cqa_scores, model_name, "cqadupstack", "cqadupstack", metric)
    cqa = np.mean(np.vstack(cqa_values), axis=0)
    arrays.append(cqa)
    arrays = np.vstack(arrays)
    means = np.mean(arrays, axis=0).round(2)
    std = np.std(arrays, axis=0).round(2)
    means_annot = []
    std_annot = []

    for idx, value in enumerate(means):
        if idx == 0:
            means_annot.append("_")
            std_annot.append("")
            continue
        means_annot.append(value)
        std_annot.append(f"±{std[idx]}")

    means_annot = np.array(means_annot).reshape(5, 6)
    std_annot = np.array(std_annot).reshape(5, 6)
    df = pd.DataFrame(
        means.reshape(5, 6),
        columns=[1024, 512, 256, 128, 64, 32],
        index=[16, 8, 4, 2, 1],
    )
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        df,
        annot=False,
        fmt="",
        cmap="crest_r",
    )

    # Get the colorbar and make it larger
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(f"{METRICS[metric]} in %", fontsize=10)

    sns.heatmap(
        df,
        annot=means_annot,
        annot_kws={"va": "bottom", "fontsize": 14},
        fmt="",
        cmap="crest_r",
        cbar=False,
    )
    sns.heatmap(
        df,
        annot=std_annot,
        annot_kws={"va": "top", "fontsize": 13},
        fmt="",
        cmap="crest_r",
        cbar=False,
    )
    plt.xlabel("Dimensionality", fontsize=14, labelpad=10)
    plt.ylabel("Number of Bits per Dimension", fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace(".pdf", ".png"), dpi=300)
    plt.close()


@click.command()
@click.argument("model_name", type=click.Choice(os.listdir(SHARE_RESULTS_FOLDER)))
def heatmap(model_name: str):
    """
    Visualize the relevance composition of the model's results.
    """
    for metric in METRICS:

        for document_length, corpus_sizes in DATASETS.items():

            for corpus_size in corpus_sizes:

                # Collect data from the model directory
                data = collect_data(model_name, document_length, corpus_size, metric)

                # Plot the heatmap chart
                plot_heatmap(data, model_name, document_length, corpus_size, metric)

        # plot_combined_heatmaps(model_name, metric)
        plot_aggregated_heatmap(model_name, metric, DATASETS)

    click.echo("Done")
