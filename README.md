# CoRECT: A Framework for Evaluating Embedding Compression Techniques

Experiments on the robustness of embedding compression methods comparing dimensionality reduction and vector quantization.

## Install Dependencies

There are two ways you can install the dependencies to run the code.

### Using Poetry (recommended)

If you have the [Poetry](https://python-poetry.org/) package manager for Python installed already, you can simply set up everything with:

```console
poetry install
source $(poetry env info --path)/bin/activate
```

After the installation of all dependencies, you will end up in a new shell with a loaded venv. In this shell, you can run the main `corect` command. You can exit the shell at any time with `exit`.

```console
corect --help
```

To install new dependencies in an existing poetry environment, you can run the following commands with the shell environment being activated:

```console
poetry lock
poetry install
```

### Using Pip (alternative)

You can also create a venv yourself and use `pip` to install dependencies:

```console
python3 -m venv venv
source venv/bin/activate
pip install .
```

## Development

### Run Code Formatting

To run the code formatting, you can use the following command:

```console
isort .
black .
```

The order of the commands is important. `isort` will sort the imports in the files, and `black` will format the code.

## Run Evaluation Code

The evaluation code currently supports two datasets: A transformed version of the MS MARCO v2 dataset, called CoRE, and public BEIR datasets.
In addition to the dataset, the code also loads an embedding model (currently Jina V3 or E5-Multilingual) to evaluate the defined compression techniques.
To start the evaluation, execute the command

```console
corect evaluate jina core     # Evaluates Jina V3 on CoRE
corect evaluate e5 beir       # Evaluates E5-Multilingual on BEIR
```

After running the evaluation code, you will find the results in the `results` folder. The results are stored in a JSON file with the name of the model and the dataset. To share the results, copy the respective JSON file to the `share_results` folder. The results are stored in the following format:

```json
{
    "ndcg_at_1": 0.38462,
    "ndcg_at_3": 0.33752,
    "ndcg_at_5": 0.30636,
    "ndcg_at_10": 0.24977,
    "ndcg_at_20": 0.31123,
    "ndcg_at_100": 0.51075,
    "ndcg_at_200": 0.55959,
    "ndcg_at_300": 0.56132,
    "ndcg_at_500": 0.56132,
    "ndcg_at_1000": 0.56132,
    "map_at_1": 0.03846,
    "map_at_3": 0.08077,
    "map_at_5": 0.10708,
    "map_at_10": 0.1392,
    "map_at_20": 0.17026,
    "map_at_100": 0.23058,
    "map_at_200": 0.24235,
    "map_at_300": 0.24262,
    "map_at_500": 0.24262,
    "map_at_1000": 0.24262,
    "recall_at_1": 0.03846,
    "recall_at_3": 0.09692,
    "recall_at_5": 0.14154,
    "recall_at_10": 0.21385,
    "recall_at_20": 0.32462,
    "recall_at_100": 0.84308,
    "recall_at_200": 0.99385,
    "recall_at_300": 1.0,
    "recall_at_500": 1.0,
    "recall_at_1000": 1.0,
    "precision_at_1": 0.38462,
    "precision_at_3": 0.32308,
    "precision_at_5": 0.28308,
    "precision_at_10": 0.21385,
    "precision_at_20": 0.16231,
    "precision_at_100": 0.08431,
    "precision_at_200": 0.04969,
    "precision_at_300": 0.03333,
    "precision_at_500": 0.02,
    "precision_at_1000": 0.01,
    "mrr_at_1": 0.38462,
    "mrr_at_3": 0.48462,
    "mrr_at_5": 0.50385,
    "mrr_at_10": 0.51581,
    "mrr_at_20": 0.52769,
    "mrr_at_100": 0.52923,
    "mrr_at_200": 0.52923,
    "mrr_at_300": 0.52923,
    "mrr_at_500": 0.52923,
    "mrr_at_1000": 0.52923,
    "rc_at_1": {
        "relevant": 0.38462,
        "distractor": 0.61538
    },
    "rc_at_3": {
        "relevant": 0.96923,
        "distractor": 2.03077
    },
    "rc_at_5": {
        "relevant": 1.41538,
        "distractor": 3.58462
    },
    "rc_at_10": {
        "relevant": 2.13846,
        "distractor": 7.83077
    },
    "rc_at_20": {
        "relevant": 3.24615,
        "distractor": 16.69231
    },
    "rc_at_100": {
        "relevant": 8.43077,
        "distractor": 88.69231
    },
    "rc_at_200": {
        "relevant": 9.93846,
        "distractor": 98.10769
    },
    "rc_at_300": {
        "relevant": 10.0,
        "distractor": 98.83077
    },
    "rc_at_500": {
        "relevant": 10.0,
        "distractor": 99.36923
    },
    "rc_at_1000": {
        "relevant": 10.0,
        "distractor": 99.63077
    }
}
```

## Extend CoRECT

### Add New Compression Technique

The currently implemented compression techniques can be found in the [quantization](src/corect/quantization) folder.
To add a new method, implement a class that extends [AbstractCompression](src/corect/quantization/AbstractCompression.py) and add your custom compression method via the `compress()` method.
To include your class in the evaluation, modify the [compression registry](src/corect/compression_registry.py) and register your class with the compression methods dictionary.
You should now be able to evaluate your compression technique by running the evaluation script as described above.

### Add New Model

New models can be added by implementing the [AbstractModelWrapper](src/corect/model_wrappers/AbstractModelWrapper.py) class, which allows you to customize the query and corpus embedding process.
The wrapper then needs to be registered in the `_get_model_wrapper()` method of the [evaluation script](src/corect/cli/evaluate.py) and the model name defined there can then be used to evaluate the model.

```
corect evaluate <model_name> core
```

### Add New Dataset

Our framework supports the addition of any HuggingFace retrieval datasets with corpus, queries and qrels splits.
To add a custom dataset, navigate to the [dataset utils](src/corect/dataset_utils.py) script, add a load function for your new dataset and register it in the `load_data()` function.
You also need to add information on the new dataset to the `datasets` dictionary in this class in the form of `datasets[<dataset_name>]=[<dataset_name>]`.
Running the evaluation script on the new dataset can then be achieved by executing the evaluation command.

```
corect evaluate jina <dataset_name>
```
