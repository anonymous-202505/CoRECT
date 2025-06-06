import json
import os
from collections import defaultdict

import click
import matplotlib.pyplot as plt

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
}
COMPRESSION_PAIRS_32 = [
    (1024, "32"),
    (1024, "1_median"),
    (512, "2_percentile"),
    (256, "4_percentile"),
    (128, "8_percentile"),
    (64, "16_casting"),
    (32, "32"),
]
COMPRESSION_PAIRS_DIM = [
    (1024, "32"),
    (512, "32"),
    (256, "32"),
    (128, "32"),
    (64, "32"),
    (32, "32"),
]
COMPRESSION_PAIRS_Q = [
    (1024, "32"),
    (1024, "16_casting"),
    (1024, "8_percentile"),
    (1024, "4_percentile"),
    (1024, "2_percentile"),
    (1024, "1_median"),
]


def collect_data(
    model_name: str, document_length: str, corpus_sizes: list, metric: str
):
    results_path = os.path.join(
        SHARE_RESULTS_FOLDER,
        model_name,
        document_length,
    )

    results = {
        "32": defaultdict(list),
        "dim": defaultdict(list),
        "q": defaultdict(list),
    }

    for dim, q in COMPRESSION_PAIRS_32:

        dim_dir = f"dim={dim}"
        q_dir = f"q={q}"
        for corpus_size in corpus_sizes:
            file_path = os.path.join(
                results_path, dim_dir, q_dir, f"{corpus_size}.json"
            )

            with open(file_path, "r") as f:
                data = json.load(f)
                value = data[metric]
                results["32"][(dim, q)].append(value)

    for dim, q in COMPRESSION_PAIRS_DIM:

        dim_dir = f"dim={dim}"
        q_dir = f"q={q}"
        for corpus_size in corpus_sizes:
            file_path = os.path.join(
                results_path, dim_dir, q_dir, f"{corpus_size}.json"
            )

            with open(file_path, "r") as f:
                data = json.load(f)
                value = data[metric]
                results["dim"][(dim, q)].append(value)

    for dim, q in COMPRESSION_PAIRS_Q:

        dim_dir = f"dim={dim}"
        q_dir = f"q={q}"
        for corpus_size in corpus_sizes:
            file_path = os.path.join(
                results_path, dim_dir, q_dir, f"{corpus_size}.json"
            )

            with open(file_path, "r") as f:
                data = json.load(f)
                value = data[metric]
                results["q"][(dim, q)].append(value)

    return results


def plot_lines(
    data_dict: dict,
    model_name: str,
    document_length: str,
    corpus_sizes: list,
    metric: str,
):
    plot_path = os.path.join(
        RESOURCES_FOLDER,
        "line_charts",
        model_name,
        metric,
        document_length,
        "{}.pdf",
    )
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    for data_index, data in data_dict.items():

        plt.figure(figsize=(6, 5))
        for (dim, q), values in data.items():
            plt.plot(corpus_sizes, values, marker="o", label=f"dim={dim}, q={q}")

        plt.xscale("log")
        # plt.xlabel("Corpus Size", fontsize=12)
        # plt.ylabel(METRICS[metric], fontsize=12)
        # plt.title(
        #     f"{METRICS[metric]} vs Corpus Size (Compression Ratio = 32)",
        #     fontsize=16,
        #     pad=20,
        # )
        plt.xticks(
            corpus_sizes,
            [f"{corpus_size:,}" for corpus_size in corpus_sizes],
            rotation=30,
            fontsize=14,
        )
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower left", fontsize=10)
        plt.tight_layout()
        plt.savefig(
            plot_path.format(data_index),
            format="pdf",
        )
        plt.close()


def plot_combined_lines(model_name: str, metric: str):
    corpus_passage = DATASETS["passage"]
    corpus_document = DATASETS["document"]

    # Collect relevant data
    data_passage = collect_data(model_name, "passage", corpus_passage, metric)
    data_document = collect_data(model_name, "document", corpus_document, metric)

    # Create subplots with shared y-axis
    _, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 5),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 0.8]},
    )

    titles = [
        "(a) Passage Retrieval on Full Precision Embeddings",
        "(b) Passage Retrieval on Quantized Embeddings",
        "(c) Document Retrieval on Quantized Embeddings",
    ]
    legend_titles = [
        "Dimensionality",
        "Number of bits\nper dimension",
        "Number of bits\nper dimension",
    ]
    keys = ["dim", "q", "q"]
    data_sources = [data_passage, data_passage, data_document]
    corpora = [corpus_passage, corpus_passage, corpus_document]

    def get_label(index, dim, q):
        if index == 0:
            return dim
        else:
            return q.split("_")[0]

    for i, ax in enumerate(axes):
        key = keys[i]
        data = data_sources[i]
        corpus_sizes = corpora[i]

        for (dim, q), values in data[key].items():
            if dim == 1024 and q == "32":
                ax.plot(
                    corpus_sizes,
                    values,
                    marker="o",
                    label=get_label(i, dim, q),
                    linewidth=4,
                    markersize=10,
                )
            else:
                ax.plot(corpus_sizes, values, marker="o", label=get_label(i, dim, q))

        ax.set_xscale("log")
        ax.set_xticks(corpus_sizes)
        ax.set_xticklabels([f"{x:,}" for x in corpus_sizes], rotation=30, fontsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(titles[i], fontsize=12.5, pad=12)
        ax.grid(True, linestyle="--", alpha=0.6)

        legend = ax.legend(title=legend_titles[i], loc="lower left", fontsize=10)
        legend.get_title().set_ha("center")
        legend.get_title().set_fontsize(10)

    plt.tight_layout()

    output_path = os.path.join(
        RESOURCES_FOLDER, "line_charts", model_name, metric, "combined.pdf"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.savefig(output_path.replace(".pdf", ".png"), dpi=300)
    plt.close()


@click.command()
@click.argument("model_name", type=click.Choice(os.listdir(SHARE_RESULTS_FOLDER)))
def line_chart(model_name: str):
    """
    Visualize the relevance composition of the model's results.
    """
    for metric in METRICS:

        # Plot combined line charts
        plot_combined_lines(model_name, metric)

        for document_length, corpus_sizes in DATASETS.items():

            # Collect data from the model directory
            data = collect_data(model_name, document_length, corpus_sizes, metric)

            # Plot the line chart
            plot_lines(data, model_name, document_length, corpus_sizes, metric)

    click.echo("Done")
