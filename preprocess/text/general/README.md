# Text Preprocess

## Common Tokens

`<unk>` for unknown word

`<eos>` for end of sentence

`<pad>` for padding, and the index is often set as `0`

## Tokenizers

## Encoder

Word encoder

```python
itos = [o for o,c in cnt.most_common()]
itos.insert(0,'<pad>')
```

### spaCy

Install English language pack

```
python -m spacy download en

conda install -c conda-forge spacy

python -m spacy download en_core_web_sm

python -m spacy link en_core_web_sm en
```
