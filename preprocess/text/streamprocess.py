# %%
import collections
import itertools


# %%
def tap(method, stream):
    return map(lambda x: (x, method(x)), stream)


def get_word_list_stream(sent_stream):
    word_list_stream = map(lambda s: s.split(), sent_stream)
    return word_list_stream


def get_word_stream(sent_stream):
    word_list_stream = map(lambda s: s.split(), sent_stream)
    word_stream = itertools.chain.from_iterable(word_list_stream)
    return word_stream


# %%
sents = ['An apple.', 'A cup of tea.']

word_lists = [wl for wl in get_word_list_stream(sents)]
words = [w for w in get_word_stream(sents)]

print(word_lists)
print(words)

counter = collections.Counter()


def update_counter(x):
    counter.update(x)


word_list_stream = tap(update_counter, map(lambda s: s.split(), sents))
word_lists = list(word_list_stream)

print(counter)
# %%
