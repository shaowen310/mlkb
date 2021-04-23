# Predict Next Element Given a Sequence

## Task

Language modeling

A language model learns to predict the probability of a sequence of words.

Training tasks proposed for language modeling:

N-gram

Given a continuous sequence and a window size \(n\). Slide the window over the whole document continuously, and use the first \(n-1\) tokens in the window to predict the last token.

Problem defined in PyTorch language modeling example

For a sequence, consecutively use first 1, 2 up to n-1 tokens to predict the next token. The stride is n-1. Don't understand why use stride.

## Data Preprocess

Append every sentence with `<eos>` token to let model learn the end of a sentence.

Question: if the punctuation "." is kept in the tokens, do we still need to insert `<eos>` token?

Needs more examples to answer the question.

The `<eos>` token used in PyTorch language modeling example means "end of paragraph".

## Model

Seqeunce -> Embedding -> Label

<- Loss

## Loss

CrossEntropyLoss

## Evaluation Metrics

Accuracy
