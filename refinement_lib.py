import re
import numeric_lib

# Helper functions that used for:
#   Text vectorization
#   Text refinement
#   Vocabulary extraction


def make_all_lower(words):
    """
    Makes all words non-capital.w
    :param words: A list of words.
    :return: Words, all non-capital.
    """
    return [word.lower() for word in words]


def has_digit(any_string):
    """
    Checks if the string has digit
    :param any_string: any string
    :return: True or False
    """
    if any(re.findall(r"[\d]+", any_string)):
        return True
    else:
        return False


def remove_stop_words(words):
    """
    Specify stop words and remove them in text.
    :param words: The list of words to be checked.
    :return: List of refined words.
    """

    stop_words = ['ve', 'lik', 'lık', 'bu', 'şu', 'o', 'ya', 'da', 'de',
                'ya da', 'her', 'şey', 'sey', 'hiç', 'bi', 'bir', 'gibi', 'daha', 'veya',
                'dahi', 'birşey', 'birsey', 'hersey', 'her şey', 'mi', 'mu', 'mü', 'ile',
                'mı', 'ise', 'ne', 'yani', 'çok', 'birçok', 'hep', 'tüm', 'den', 'dan', 'tum',
                'sana', 'bize', 'size', 'sen', 'onlar', 'li', 'lı', 'kadar', 'çok', 'cok',
                'sonra', 'icin', 'için']
    ind = 0
    while ind < len(words):
        if words[ind] in stop_words:
            words.remove(words[ind])
        else:
            ind = ind + 1

    return words


def remove_digits(words):
    """
    If the word includes digits, remove them from text.
    :param words: The list of words to check.
    :return: List of refined words.
    """
    ind = 0
    while ind < len(words):
        if has_digit(words[ind]):
            words.remove(words[ind])
        else:
            ind = ind + 1

    return words


def remove_single_lettereds(words):
    """
    If the word includes short words, remove them.
    :param words: The list of words to check.
    :return: List of refined words.
    """
    ind = 0
    while ind < len(words):
        if len(words[ind]) < 2:
            words.remove(words[ind])
        else:
            ind = ind + 1
    return words


def tokenization(words):
    """
        Some tokenization stuff here:
        Avoid stop words and empty strings and digits etc
        :param words: A list of words to refine.
        :return: also words, refined.
    """
    no_stop_words = remove_stop_words(words)
    no_digits = remove_digits(no_stop_words)
    no_single_lettereds = remove_single_lettereds(no_digits)

    return no_single_lettereds


def is_including(any_string, punctuation):
    """
    Checks if any any_string includes punctuation.
    :param any_string: String to check.
    :param punctuation: Punc. to search for.
    :return: True, False.
    """
    if punctuation in any_string:
        return True
    else:
        return False


def get_words(any_string):
    """
    Get the words.
    :param any_string:
    :return: Words of strings.
    """
    #   Check for the apostrophe and remove it
    if not is_including(any_string, '\''):
        return re.findall(r"[\w']+", any_string)
    else:
        return get_words(''.join(any_string.split("'")))


def vectorize(sentence, indicators):
    """
    Get vectors of each instance, according to the presence of the indicator vocabulary.
    Vocabularies of spams and not spams.
    :param sentence:
    :param indicators:
    :return: vector of binary values.
    """
    words = make_all_lower(tokenization(get_words(sentence)))
    vector = numeric_lib.zeros(len(indicators))
    for i in range(len(words)):
        if words[i] in indicators.keys():
            vector[i] = 1

    return vector
