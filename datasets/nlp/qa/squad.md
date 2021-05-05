## SQuAD

Official page https://rajpurkar.github.io/SQuAD-explorer/

### V1.1

train https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json

dev https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json

```shell
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O "train-v1.1.json"

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O "dev-v1.1.json"
```

huggingface

description https://huggingface.co/datasets/squad

code https://github.com/huggingface/datasets/tree/master/datasets/squad

```python
from datasets import load_dataset

dataset = load_dataset("squad")
```

cite

```
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
```

### V2.0

Download link https://rajpurkar.github.io/SQuAD-explorer/

```shell
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O "train-v2.0.json"

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O "dev-v2.0.json"
```
