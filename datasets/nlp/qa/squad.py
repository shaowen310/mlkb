from datasets import load_dataset

dataset = load_dataset("squad")

_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
_URLS = {
    "train": _URL + "train-v1.1.json",
    "dev": _URL + "dev-v1.1.json",
}
