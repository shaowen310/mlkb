import argparse
import os
import json

import torch

from transformers import PretrainedConfig
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME

from logging_utils import get_logger

_logger = get_logger(__name__)


def save_model(model, output_dir, tokenizer=None, training_args=None):
    os.makedirs(output_dir, exist_ok=True)
    _logger.info(f"Saving model checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    if training_args is not None:
        with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
            json.dump(training_args.__dict__, f)


def load_training_args(dir):
    with open(os.path.join(dir, "training_args.json"), 'r') as f:
        parser = argparse.ArgumentParser()
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        return parser.parse_args('', namespace=t_args)


def load_model_config(dir):
    _logger.info(f"Loading model config from {dir}).")

    return PretrainedConfig.from_json_file(os.path.join(dir, CONFIG_NAME))


def load_model_weights(model, dir):
    _logger.info(f"Loading model weights from {dir}).")

    if os.path.isfile(os.path.join(dir, CONFIG_NAME)):
        # We load the model state dict on the CPU to avoid an OOM error.
        state_dict = torch.load(os.path.join(dir, WEIGHTS_NAME), map_location="cpu")
        # If the model is on the GPU, it still works!
        load_result = model.load_state_dict(state_dict, strict=False)
        if len(load_result.missing_keys) != 0:
            if load_result.missing_keys == model._keys_to_ignore_on_save:
                model.tie_weights()
            else:
                _logger.warn(
                    f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}."
                )
        if len(load_result.unexpected_keys) != 0:
            _logger.warn(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )
