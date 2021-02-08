import os
import torch


class WhitespaceTokenizer:
    def tokenize(sentence):
        return sentence.split()
