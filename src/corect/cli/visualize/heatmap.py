import json
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from corect.config import *

METRICS = {
    "ndcg_at_10": "NDCG@10",
    "recall_at_100": "Recall@100",
    "recall_at_200": "Recall@200",
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
    # "arguana": ["arguana"],
    # "climate-fever": ["climate-fever"],
    # "cqadupstack-android": ["cqadupstack-android"],
    # "cqadupstack-english": ["cqadupstack-english"],
    # "cqadupstack-gaming": ["cqadupstack-gaming"],
    # "cqadupstack-gis": ["cqadupstack-gis"],
    # "cqadupstack-mathematica": ["cqadupstack-mathematica"],
    # "cqadupstack-physics": ["cqadupstack-physics"],
    # "cqadupstack-programmers": ["cqadupstack-programmers"],
    # "cqadupstack-stats": ["cqadupstack-stats"],
    # "cqadupstack-tex": ["cqadupstack-tex"],
    # "cqadupstack-unix": ["cqadupstack-unix"],
    # "cqadupstack-webmasters": ["cqadupstack-webmasters"],
    # "cqadupstack-wordpress": ["cqadupstack-wordpress"],
    # "dbpedia": ["dbpedia"],
    # "fever": ["fever"],
    # "fiqa": ["fiqa"],
    # "hotpotqa": ["hotpotqa"],
    # "nfcorpus": ["nfcorpus"],
    # "nq": ["nq"],
    # "quora": ["quora"],
    # "scidocs": ["scidocs"],
    # "scifact": ["scifact"],
    # "touche2020": ["touche2020"],
    # "trec-covid": ["trec-covid"],
}


def collect_data(model_name: str, document_length: str, corpus_size: int, metric: str):
    results_path = os.path.join(
        SHARE_RESULTS_FOLDER,
        model_name,
        document_length,
    )

    heatmap_data = {}
    for dim_dir in os.listdir(results_path):
        dim_path = os.path.join(results_path, dim_dir)
        if not os.path.isdir(dim_path) or not dim_dir.startswith("dim="):
            continue

        dim = int(dim_dir.split("=")[1])
        heatmap_data[dim] = {}

        for q_dir in os.listdir(dim_path):
            q_path = os.path.join(dim_path, q_dir)
            if not os.path.isdir(q_path) or not q_dir.startswith("q="):
                continue

            q_method = q_dir.split("=")[1]
            q = int(q_method.split("_")[0])
            json_file = os.path.join(q_path, f"{corpus_size}.json")

            if os.path.isfile(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                    m = data.get(metric)
                    if m is not None:
                        heatmap_data[dim][q] = m * 100

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": f"{METRICS[metric]} in %"},
    )
    # plt.title(
    #     f"Heatmap of {METRICS[metric]} in % ({document_length.capitalize()}, Corpus Size: {corpus_size:,})",
    #     fontsize=16,
    #     pad=20,
    # )
    plt.xlabel("Number of Bits per Dimension", fontsize=14, labelpad=10)
    plt.ylabel("Dimensionality", fontsize=14, labelpad=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(
        plot_path,
        format="pdf",
    )
    plt.close()


def plot_combined_heatmaps(model_name: str, metric: str):
    plots = [
        ("passage", 10_000),
        ("passage", 100_000_000),
        ("document", 10_000),
        ("document", 10_000_000),
    ]

    data_dicts = []
    vmin, vmax = float("inf"), float("-inf")

    # Step 1: Collect all data & determine color scale range
    for document_length, corpus_size in plots:
        data = collect_data(model_name, document_length, corpus_size, metric)
        data_dicts.append(data)

        df = pd.DataFrame(data).T
        vmin = min(vmin, df.min().min())
        vmax = max(vmax, df.max().max())

    # Step 2: Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(14, 6), sharey=True)

    for ax, (document_length, corpus_size), data in zip(axes, plots, data_dicts):
        df = (
            pd.DataFrame(data)
            .T.sort_index(ascending=False)
            .sort_index(axis=1, ascending=False)
        )
        sns.heatmap(
            df,
            ax=ax,
            annot=True,
            annot_kws={"fontsize": 16},
            fmt=".2f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
        )
        ax.set_title(
            f"{document_length.capitalize()}, Corpus: {corpus_size:,}", fontsize=16
        )
        ax.set_xlabel("Number of Bits per Dimension", fontsize=16)
        ax.set_ylabel("Dimensionality" if ax == axes[0] else "", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)

    # Step 3: Add one shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=f"{METRICS[metric]} in %")

    plt.suptitle(
        f"{METRICS[metric]} in %",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave room for colorbar and title

    # Save
    save_path = os.path.join(
        RESOURCES_FOLDER, "heatmaps", model_name, metric, "combined.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format="pdf")
    plt.close()


def plot_aggregated_heatmap(model_name: str, metric: str, datasets: dict):
    arrays, cqa_values = [], []
    cqa_scores = {}
    save_path = os.path.join(
        RESOURCES_FOLDER, "heatmaps", model_name, metric, "aggregated.pdf"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for document_length, corpus_sizes in datasets.items():
        for corpus_size in corpus_sizes:
            # Collect data from the model directory
            data = collect_data(model_name, document_length, corpus_size, metric)
            full_prec_value = data[1024][32]
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

    for dim, scores in cqa_scores.items():
        for q, score in scores.items():
            cqa_scores[dim][q] = np.mean(cqa_scores[dim][q])

    plot_heatmap(cqa_scores, model_name, "cqadupstack", "cqadupstack", metric)
    cqa = np.mean(np.vstack(cqa_values), axis=0)
    arrays.append(cqa)
    arrays = np.vstack(arrays)
    means = np.mean(arrays, axis=0).round(2)
    std = np.std(arrays, axis=0).round(2)
    means_annot, std_annot = [], []

    for idx, value in enumerate(means):
        means_annot.append(value)
        std_annot.append(f"±{std[idx]}")

    means_annot = np.array(means_annot).reshape(6, 6)
    std_annot = np.array(std_annot).reshape(6, 6)
    df = pd.DataFrame(
        means.reshape(6, 6),
        columns=[32, 16, 8, 4, 2, 1],
        index=[1024, 512, 256, 128, 64, 32],
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=False, fmt="", cmap="viridis")
    sns.heatmap(
        df,
        annot=means_annot,
        annot_kws={"va": "bottom", "fontsize": 16},
        fmt="",
        cmap="viridis",
        cbar=False,
    )
    sns.heatmap(
        df,
        annot=std_annot,
        annot_kws={"va": "top", "fontsize": 14},
        fmt="",
        cmap="viridis",
        cbar=False,
    )
    plt.xlabel("Number of Bits per Dimension", fontsize=14, labelpad=10)
    plt.ylabel("Dimensionality", fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf")
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

        plot_combined_heatmaps(model_name, metric)
        plot_aggregated_heatmap(model_name, metric, DATASETS)

    click.echo("Done")
