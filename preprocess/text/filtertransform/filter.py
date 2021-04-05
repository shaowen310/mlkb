import string


def remove_punc(token: str):
    return token.translate(str.maketrans('', '', string.punctuation))


def remove_digits(token: str):
    return token.translate(str.maketrans('', '', string.digits))
