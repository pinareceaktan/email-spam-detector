import csv
import random
import re
from subprocess import call
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--use_sckit_learn', type=bool, default=True,
                    help='Use sckitlearn.normalize.scale method')


def create_list_of_words(words):
    return [word.lower() for word in words]


def has_digit(any_string):
    if any(re.findall(r"[\d]+", any_string)):
        return True
    else:
        return False


def zeros(n):
    listofzeros = [0] * n
    return listofzeros


def remove_stop_words(words):

    stop_words = ['ve', '', '.', ',', 'lik', 'lık', 'bu', 'şu', 'o', 'ya', 'da', 'de',
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
    ind = 0
    while ind < len(words):
        if has_digit(words[ind]):
            words.remove(words[ind])
        else:
            ind = ind + 1

    return words


def remove_single_lettereds(words):
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
        :param words:
        :return: also words
    """
    no_stop_words = remove_stop_words(words)
    no_digits = remove_digits(no_stop_words)
    no_single_lettereds = remove_single_lettereds(no_digits)

    return no_single_lettereds


def make_all_lower(words):
    lower_words = []
    for word in words:
        lower_words.append(word.lower())
    return lower_words


def isIncludes(anysString, punctuation):
    if punctuation in anysString:
        return True
    else:
        return False


def get_words(anysString):
    if not isIncludes(anysString, '\''):
        return re.findall(r"[\w']+", anysString)
    else:
        return get_words(''.join(anysString.split("'")))


def get_punctuations(any_string):
    return re.findall(r"[^ \w\t\n\r\f\v]+", any_string)


def get_digits(any_string):
    return re.findall(r"[\d]+", any_string)


def get_stop_words(words):
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
    return re.findall(r"\b[a-zA-Z]{2}\b", any_string)


def get_capitals(any_string):
    return re.findall(r"([A-Z]+[a-z]*)", any_string)


def get_avg_word_length(words):
    lengths = 0
    for word in words:
        lengths = len(word) + lengths
    return int(lengths / len(words))


def count_spaces(any_string):
    return len(re.findall(r"[ \t\n\r\f\v]", any_string))


def vectorize(sentence, indicators):
    words = make_all_lower(tokenization(get_words(sentence)))
    vector = zeros(len(indicators))
    for i in range(len(words)):
        if words[i] in indicators.keys():
            vector[i] = 1

    return vector


def normalization(vector):
    return [((x - min(vector)) / (max(vector) - min(vector))) for x in vector]


def add_dimension(vector):
    """ If the ndarray sized (n,) adds second dimension
    makes all (n, 1) sized tuple
    :param
        vector: Any ndarray sized (n,)
    :return
        vector : ndarray sized (n,1)
    """
    return np.array([[node] for node in vector])


def accuracy(predictions, ground_truth):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == ground_truth[i]:
            correct = correct +1
    return correct, correct * 100 / len(predictions)


def derivative(x):
    return x * (1.0 - x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def feed_forward(features, weight_0, weight_1):
    """
    Computes the final layer of network model, forward propagates the network.
    :param features: Data features to perform evaluations
    :param weight_0: 1.st weights
    :param weight_1: 2.nd weights
    :return: layer_2:
    """
    layer_0 = features
    layer_1 = sigmoid(np.dot(layer_0, weight_0))
    layer_2 = sigmoid(np.dot(layer_1, weight_1))
    return layer_2


def threshold_network(final_layer):
    """
    Threshold the network: If the output fires the activation label it as 1 and 0 otherwise
    :param final_layer:
    :return: test_predictions in a binary format
    """
    test_predictions = []

    for i in range(len(final_layer)):
        if final_layer[i][0] > 0.5:
            test_predictions.append(1)
        else:
            test_predictions.append(0)
    return test_predictions


def main(argv=None):

    with open('data.csv', newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=';')
        sentences = {}
        for row in spamreader:
            sentences[row[0]] = row[1]

        # Train, validation, test data partition
        train_size = int(len(sentences) * .7)
        validation_size = int(len(sentences) * .1)

        # Arrange train and test data
        sentences_list = []
        for sentence, value in sentences.items():
            sentences_list.append([sentence, value])

        random.shuffle(sentences_list)

        # These are raw data, not features, it is now required to extract features
        train_data = sentences_list[:train_size]
        #   validation_data = sentences_list[train_size:train_size+validation_size]
        #   test_data = sentences_list[train_size+validation_size:]

        # Create vocabulary for each class, use train data
        spam_vocab_dict = dict()
        not_spam_vocab_dict = dict()

        for sentence in train_data:
            words = make_all_lower(tokenization(get_words(sentence[0])))
            for word in words:
                if sentence[1] == "spam":
                    if word not in spam_vocab_dict.keys():
                        spam_vocab_dict[word] = 0
                elif sentence[1] == "normal":
                    if word not in not_spam_vocab_dict.keys():
                        not_spam_vocab_dict[word] = 0

        for sentence_entry in train_data:
            sentence = sentence_entry[0]
            label = sentence_entry[1]
            words = make_all_lower(tokenization(get_words(sentence)))
            if label == "spam":
                for word in words:
                    spam_vocab_dict[word] = spam_vocab_dict[word] + 1
            elif label == "normal":
                for word in words:
                    not_spam_vocab_dict[word] = not_spam_vocab_dict[word] + 1

        # Rank the vocabularies
        spam_indicators = dict(sorted(spam_vocab_dict.items(), key=lambda x: x[1], reverse=True)[:45])
        not_spam_indicators = dict(sorted(not_spam_vocab_dict.items(), key=lambda x: x[1], reverse=True)[:45])

        features = []
        labels = []
        # Create features and labels from sentences
        for sentence in sentences_list:
            feature_sub_set = []

            # 1. Count of words in the text
            feature_sub_set.append(len(get_words(sentence[0])))
            # 2. Punctuation count in text
            feature_sub_set.append(len(get_punctuations(sentence[0])))
            # 3. Digit count in the text
            feature_sub_set.append(len(get_digits(sentence[0])))
            # 4. Stop word counts in the text
            feature_sub_set.append(len(get_stop_words(get_words(sentence[0]))))
            # 5. Count of 2 lettered words in the text
            feature_sub_set.append(len(get_short_words(sentence[0])))
            # 6. Count of words that include capitals in the text
            feature_sub_set.append(len(get_capitals(sentence[0])))
            # 7. Average word len in text
            feature_sub_set.append(get_avg_word_length(get_words(sentence[0])))
            # 8. Count of white spaces in text
            feature_sub_set.append(count_spaces(sentence[0]))
            # 9. Create sparse vectors of spam indicator presences sized 45
            feature_sub_set.extend(vectorize(sentence[0], spam_indicators))
            # 10. Create sparse vectors of not spam indicator presences sized 45
            feature_sub_set.extend(vectorize(sentence[0], not_spam_indicators))

            features.append(feature_sub_set)

            labels.append(1 if sentence[1] == 'spam' else 0)

        # Now get train features and labels
        train_features = features[0:train_size]
        train_labels = labels[0:train_size]
        validation_features = features[train_size:train_size+validation_size]
        validation_labels = labels[train_size:train_size+validation_size]
        test_features = features[train_size+validation_size:]
        test_labels = labels[train_size+validation_size:]

        # Normalization stuff here
        if FLAGS.use_sckit_learn:
            train_features = preprocessing.scale(train_features)
            test_features = preprocessing.scale(test_features)
            validation_features = preprocessing.scale(validation_features)
            train_labels = add_dimension(np.array(train_labels))
            test_labels = np.array(test_labels)
            validation_labels = np.array(validation_labels)
        else:
            train_features = np.array([normalization(feature) for feature in train_features])
            test_features = np.array([normalization(feature) for feature in test_features])
            validation_features = np.array([normalization(feature) for feature in validation_features])
            train_labels = add_dimension(np.array(train_labels))
            test_labels = np.array(test_labels)
            validation_labels = np.array(validation_labels)

        # Classifier: Multi Layer Perceptron
        # Network Parameters:
        # input layer has len(train_features) nodes
        # hidden layer has 8 nodes
        # output layer has 1 node
        _, feature_size = train_features.shape
        dim1 = feature_size
        dim2 = 8

        # Randomly initialize the weight vectors
        np.random.seed(1)
        weight_0 = 2 * np.random.random((dim1, dim2)) - 1 # 98x8
        weight_1 = 2 * np.random.random((dim2, 1)) - 1 # 8x1

        # Training loop here:
        err_log = []
        go_on_task = True
        epochs = 0

        while go_on_task:
            layer_0 = train_features
            layer_1 = sigmoid(np.dot(layer_0, weight_0))
            layer_1 = np.array(layer_1)
            layer_2 = sigmoid(np.dot(layer_1, weight_1))

            layer_2_err = train_labels - layer_2

            # perform back propagation
            layer_2_delta = layer_2_err * derivative(layer_2)
            layer_1_error = layer_2_delta.dot(weight_1.T)
            layer_1_delta = layer_1_error * derivative(layer_1)

            # update the weight vectors
            weight_1 += layer_1.T.dot(layer_2_delta)
            weight_0 += layer_0.T.dot(layer_1_delta)

            err_log.append(np.mean(layer_2_err))
            epochs = epochs + 1
            # Log error every 5 epoch
            if epochs % 5 == 0:
                print([epochs, ".epoch, error is=> ", np.mean(layer_2_err)])

            # Design an early stopping mechanism, we don not want an overfit!
            # Use std on error logs, if std is smaller between batches
            # Than the network simply started to memorize and not learning a thing
            # Use validation error also
            if epochs % 40 == 0 and epochs != 0:
                improvement = np.std((err_log[epochs - 40:epochs - 20])) - np.std((err_log[epochs - 20:epochs]))
                final_layer = feed_forward(validation_features, weight_0, weight_1)
                test_predictions = threshold_network(final_layer)
                _, score = accuracy(test_predictions, validation_labels)

                if abs(improvement) > 0.015 and score < 70:
                    go_on_task = True
                else:
                    go_on_task = False

        # evaluation on the testing data
        final_layer = feed_forward(test_features, weight_0, weight_1)
        test_predictions = threshold_network(final_layer)
        correct, score = accuracy(test_predictions, test_labels)

        # printing the output
        print("total = ", len(final_layer))
        print("correct = ", correct)
        print("accuracy = ", score)


if __name__ == '__main__':

    FLAGS = parser.parse_args()
    try:
        from sklearn import preprocessing

    except ImportError:
        print("|=====================================================================================|")
        print("| This code requires scikit-learn module to scale sparse features                     |")
        print("| Network will work better with sckit-learn normalization                             |")
        print("| But if you want to abort installation, basic 0-1 normalization will be performed    |")
        print("| Do you like to install sckit-learn now? [y/n]                                       |")
        print("|=====================================================================================|")
        if (input()).lower() == 'y':
            call(["python3", "-m", "pip", "install", "-U", "scikit-learn"])
            # Also install numpy and stuff
            call(["python3", "-m", "pip", "install", "--user", "numpy", "scipy", "matplotlib", "ipython", "jupyter", "pandas", "sympy", "nose"])
            from sklearn import preprocessing

        else:
            FLAGS.use_sckit_learn = 0
    main()


