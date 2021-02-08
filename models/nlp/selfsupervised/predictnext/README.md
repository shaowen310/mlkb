# Predict Next Element Given a Sequence

## Preliminaries

Given a running text and we extract subsequences with window size $n$. We want to predict the last word by previous words in a subsequence.

## Models

A neural probabilistic language model (NPLM)

## Loss Functions

Cross-entropy loss

## Evaluation Metrics

Accuracy

## Tips

### Weight initialization

Embedding layer weight

```python
initrange = 0.1
nn.init.uniform_(self.encoder.weight, -initrange, initrange)
```

Why?

### Overfit

Dropout layer after

1. Embedding
2. Non-linear layer

Try larger drop rate such as 0.5

### Stop propagation for RNN-like model
