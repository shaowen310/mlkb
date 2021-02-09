# Machine Learning Practice

## Architecture

Wide -> narrow -> wide

## Models

### RNN-like

#### Stop propagation of hidden layer

```python
hidden = repackage_hidden(hidden)
```

After each batch

#### Avoid gradient explosion

```python
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
optimizer.step()
```
