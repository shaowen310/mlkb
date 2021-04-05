import unicodedata


def unicode_to_ascii(s: str):
    """Turn a Unicode string to plain ASCII, thanks to 
    http://stackoverflow.com/a/518232/2809427

    Args:
        s (str): unicode-encoded string

    Returns:
        str: ascii-encoded string
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')