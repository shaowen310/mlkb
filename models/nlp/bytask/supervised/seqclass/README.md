# Sequence Classificaton

## Task

Given sequences and labels, we want to train a model that can predict label for new sequences.

## Data Preprocess

Sequences of different lengths.

Solutions:

* Cut and padding

## Model

Seqence -> Embedding -> Label 

<- Loss

## Loss

`CrossEntropyLoss` if labels don't contain confidence level

`BCEWithLogitsLoss` if labels contain confidence level

## Evaluation

Accuracy, f1
