from nltk.stem import PorterStemmer

_porter = PorterStemmer()


def stem(token):
    return _porter.stem(token)
