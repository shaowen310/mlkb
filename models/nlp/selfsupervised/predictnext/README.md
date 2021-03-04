# Predict Next Element Given a Sequence

## Data

Should be independent!

Should not be (1, 2, 3) -> 4 and (2, 3, 4) -> 5, should be (1, 2, 3) -> 4, (4, 5, 6) -> 7

## Notes

1. Add `<eos>` token to every sentence.

## Preliminaries

Given a running text and we extract subsequences with window size $n$. We want to predict the last word by previous words in a subsequence.

## Models

A neural probabilistic language model (NPLM)

LSTM

LSTM Output -> fully connected -> log-softmax

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

Try larger dropout rate such as 0.5
