import string


def remove_punc(token):
    return token.translate(str.maketrans('', '', string.punctuation))


def remove_digits(token):
    return token.translate(str.maketrans('', '', string.digits))


def generate_transformed_tokens(tokens):
    for (t, doc) in tokens:
        t = remove_punc(t)
        t = remove_digits(t)
        if t == '':
            continue
        yield (t, doc)
