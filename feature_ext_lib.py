import re


def get_punctuations(any_string):
    """
    Get the punctuations in the string.
    :param any_string:
    :return: The list of all punctuations.
    """
    return re.findall(r"[^ \w\t\n\r\f\v]+", any_string)


def get_digits(any_string):
    """
    Get digits in the string.
    :param any_string:
    :return: The list of all digits in the string.
    """
    return re.findall(r"[\d]+", any_string)


def get_stop_words(words):
    """
    Get stop words in the string.
    :param words:
    :return: A list of stop words in the string.
    """
    stop_words = ['ve', '', '.', ',', 'lik', 'lık', 'bu', 'şu', 'o', 'ya', 'da', 'de',
                'ya da', 'her', 'şey', 'sey', 'hiç', 'bi', 'bir', 'gibi', 'daha', 'veya',
                'dahi', 'birşey', 'birsey', 'hersey', 'her şey', 'mi', 'mu', 'mü', 'ile',
                'mı', 'ise', 'ne', 'yani', 'çok', 'birçok', 'hep', 'tüm', 'den', 'dan', 'tum',
                'sana', 'bize', 'size', 'sen', 'onlar', 'li', 'lı', 'kadar', 'çok',
                'cok', 'sonra', 'icin', 'için']
    presence = []
    for word in words:
        if word in stop_words:
            presence.append(word)
    return presence


def get_short_words(any_string):
    """
    Get short words in the string.
    :param any_string:
    :return: A list of all short words in the string.
    """
    return re.findall(r"\b[a-zA-Z]{2}\b", any_string)


def get_capitals(any_string):
    """
    Get the words includes capitals.
    :param any_string:
    :return: The list ow words present capitals.
    """
    return re.findall(r"([A-Z]+[a-z]*)", any_string)


def get_avg_word_length(words):
    """
    Get average word length in the string.
    :param words:
    :return: average word length.
    """
    lengths = 0
    for word in words:
        lengths = len(word) + lengths
    return int(lengths / len(words))


def count_spaces(any_string):
    """
    Count spaces in the string.
    :param any_string:
    :return: Number of blank lines in the string.
    """
    return len(re.findall(r"[ \t\n\r\f\v]", any_string))
