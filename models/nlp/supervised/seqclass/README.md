# Sequence Classificaton

## Preliminaries

Given sequences and labels, we want to train a model that can predict label for new sequences.

## Challenges

Sequences of different lengths.

## Model

LSTM

## Loss

Cross-entropy loss if labels don't contain confidence level

BCEWithLogitsLoss if labels contain confidence level

## Extension

Multi-class problem
