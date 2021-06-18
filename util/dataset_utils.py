import os
import json

import datasets

from logging_utils import get_logger

_logger = get_logger(__name__)


def save_dataset(dataset: datasets.Dataset, output_dir, dataset_args=None):
    os.makedirs(output_dir, exist_ok=True)
    dataset.save_to_disk(output_dir)

    if dataset_args is not None:
        with open(os.path.join(output_dir, "dataset_args.json"), 'w') as f:
            json.dump(dataset_args.__dict__, f)


def load_dataset(generate_dataset_func, cache_dir=None, dataset_args=None):
    if cache_dir is not None and os.path.isdir(cache_dir):
        _logger.info(f"Loading dataset from {cache_dir}")
        return datasets.load_from_disk(cache_dir)
    else:
        dataset = generate_dataset_func()
        if cache_dir is not None:
            _logger.info(f"Saving dataset to {cache_dir}")
            save_dataset(dataset, cache_dir, dataset_args=dataset_args)
        return dataset
