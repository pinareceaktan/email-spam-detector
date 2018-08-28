import numpy as np


def mse(subtraction):
    return np.mean(subtraction ** 2)

# Naive bayes utility functions


def word_likelihood(vocabulary, alpha, sample_size):
    """ For each word in the text,
        Computes the word's likelihood for classification.
        It is used to calculate conditional probability  p(Mail's Content | Mail is a spam)
    :param vocabulary
    :param alpha: An emprical value to prevent overfit for the case of classification of a not recognized word in test.
    :param sample_size
    :returns the vocabulary
    """
    vocabulary.update((k, v + alpha / (sample_size + alpha * len(vocabulary))) for k, v in vocabulary.items())
    return vocabulary


def conditional_probability_of(given_email, is_spam, in_dict):
    """
    Conditional probabilities  p(Mail's Content | Mail is a spam)
    :param given_email:
    :param is_spam:
    :param in_dict:
    :return: result
    """
    if is_spam == 1:
        result = 1.0
        for i in range(len(given_email)):
            if given_email[i] in in_dict.keys():
                result = in_dict[given_email[i]] * result
            else:
                result = result * 0.00002

    elif is_spam == 0:
        result = 1.0
        for j in range(len(given_email)):
            if given_email[j] in in_dict.keys():
                result = in_dict[given_email[j]] * result
            else:
                result = result * 0.00002
    return result

# Perceptron utility functions:


def derivative(x):
    """
    Take the derivative of x.
    Required for back prop.
    :param x:
    :return: derivatives.
    """
    return x * (1.0 - x)


def sigmoid(x):
    """Sigmoid function"""
    try:
        return 1.0 / (1.0 + np.exp(-x))
    except RuntimeError:
        pass


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
