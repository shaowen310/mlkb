from nltk.stem import PorterStemmer

_porter = PorterStemmer()


def stem(token: str):
    return _porter.stem(token)
