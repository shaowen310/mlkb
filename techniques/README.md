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