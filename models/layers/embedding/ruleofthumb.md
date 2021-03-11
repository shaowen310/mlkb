# Rule of Thumb

## Embedding dimension

### Fixed

According to https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html

```python
embedding_dimensions =  vocab_size**0.25
```

According to https://forums.fast.ai/t/embedding-layer-size-rule/50691/13

```python
a * log(n_cat)
min(600, round(1.6 * n_cat ** .56)
```

```
300
```

### Dynamic (contextual)

Up to 1024
