# Sequence Classificaton

## Preliminaries

Given sequences and labels, we want to train a model that can predict label for new sequences.

## Challenges

Sequences of different lengths.

Solutions:

1. Cut and padding

## Model

LSTM

## Loss

Cross-entropy loss if labels don't contain confidence level

BCEWithLogitsLoss if labels contain confidence level

## Extension

Multi-class problem

BCEWithLogitsLoss
