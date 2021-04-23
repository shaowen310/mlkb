# Predict Next Element Given a Sequence

## Task

Given a continuous sequence and a window size \(n\). Slide the window over the whole document continuously, and use the first \(n-1\) tokens in the window to predict the last token.

This task is often referred to as language modeling.

## Data Preprocess

Append every sentence with `<eos>` token to let model learn the end of a sentence.

Question: if the punctuation "." is kept in the tokens, do we still need to insert `<eos>` token?

## Model

Seqeunce -> Embedding -> Label

<- Loss

## Loss

CrossEntropyLoss

## Evaluation Metrics

Accuracy
