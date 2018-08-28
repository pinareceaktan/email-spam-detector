def accuracy(predictions, ground_truth):
    """
    Returns the number of true positives and accuracy score
    :param predictions:
    :param ground_truth:
    :return: number of true positives, and score
    """
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == ground_truth[i]:
            correct = correct + 1
    return correct, correct * 100 / len(predictions)


def sensivity(predictions, ground_truth):
    """
    Number of true positives divided by whole
    :param predictions:
    :param ground_truth:
    :return: number of corrects, score
    """
    correct = 0
    for i in range(len(predictions)):
        if ground_truth[i] == 1:
            if predictions[i] == ground_truth[i]:
                correct = correct + 1
    return correct, correct * 100 / len(predictions)


def specifity(predictions, ground_truth):
    """
    Number of true negatives divided by whole
    :param predictions:
    :param ground_truth:
    :return: number of corrects, score
    """
    correct = 0
    for i in range(len(predictions)):
        if ground_truth[i] == 0:
            if predictions[i] == ground_truth[i]:
                correct = correct + 1
    return correct, correct * 100 / len(predictions)


def efficiency(sensitivity, specifity):
    """
    Efficiency.
    :param sensitivity:
    :param specifity:
    :return:
    """
    return (sensitivity + specifity) / 2



