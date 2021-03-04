import collections


class TokenCounter(collections.Counter):
    def get_vocab(special_tokens=['<pad>']):
        '''
        Returns:
            list: vocab list
        '''
        vocab = special_tokens
        vocab.extend([w for w, _ in super().most_common()])
        return vocab


if __name__ == '__main__':
    word_list = ['The', 'quick', 'brown', 'fox']
    word_count = TokenCounter(word_list)
    word_count.update(word_list)
    print(word_count)
