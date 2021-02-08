import collections


class WordCounter(collections.Counter):
    def get_vocab(insert_pad=False):
        vocab = [o for o, _ in super().most_common()]
        if insert_pad:
            vocab.insert(0, '<pad>')
        return vocab


if __name__ == '__main__':
    word_list = ['The', 'quick', 'brown', 'fox']
    word_count = WordCounter(word_list)
