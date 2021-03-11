# References
# https://www.geeksforgeeks.org/nlp-how-tokenizing-text-sentence-words-works/
# %%
import nltk

# install NLTK data
# https://www.nltk.org/data.html
# install to ~/nltk_data
# python -m nltk.downloader all

# %%
sentence = "The quick brown fox jumps over the lazy dog."

# %% [markdown]
# ## Word tokenizers
# %%
# Recommended
words = nltk.word_tokenize(sentence)
print(words)

# %%
word_tokenizer = nltk.tokenize.TreebankWordTokenizer()
words = word_tokenizer.tokenize(sentence)

# %%
ws_tokenizer = nltk.tokenize.WhitespaceTokenizer()
words = ws_tokenizer.tokenize(sentence)

# %% [markdown]
# ## Sentence tokenizers
# %%
doc = 'A sentence. Another sentence.'
# %%
# Recommended
sents = nltk.sent_tokenize(doc)
