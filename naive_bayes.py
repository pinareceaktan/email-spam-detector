import csv
import random
import numpy as np

import refinement_lib
import classifier_utils
import confusion_matrix


def main(argv=None):

    with open('data.csv', newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=';')
        sentences = {}
        for row in spamreader:
            sentences[row[0]] = row[1]

        # Create a raw list from sentences
        raw_data = []
        for sentence, value in sentences.items():
            words = refinement_lib.get_words(sentence)
            also_words = refinement_lib.tokenization(words)
            raw_data.append([refinement_lib.make_all_lower(also_words), value])

        # Train and test data partition
        train_size = int(len(sentences) * .7)

        # =========== Whole stuff here: Runs 5 times ===========================

        model_accuracies = []

        for e in range(5):
            # Arrange train and test data
            random.shuffle(raw_data)

            train_data = raw_data[:train_size]
            test_data = raw_data[train_size:]

            # Initialize bag of words model: Create Vocabulary from train data

            spam_vocabularies = dict()
            not_spam_vocabularies = dict()

            for i in range(len(train_data)):
                for word in range(len(train_data[i][0])):
                    if train_data[i][1] == "spam":
                        if (train_data[i][0][word]).lower() not in spam_vocabularies.keys():
                            spam_vocabularies[(train_data[i][0][word]).lower()] = 0
                    elif train_data[i][1] == "normal":
                        if (train_data[i][0][word]).lower() not in not_spam_vocabularies.keys():
                            not_spam_vocabularies[(train_data[i][0][word]).lower()] = 0

            # Prepare Labels
            labels = []
            for i in range(len(raw_data)):
                labels.append(1 if raw_data[i][1] == "spam" else 0)

            train_labels = labels[0:train_size]
            test_labels = labels[train_size:]

            """
            Naive bayes classifier

            p(A|B) = P(B|A)*P(A)/P(B)
            p(Given that mail is spam | Mail's Content) = p(Mail's Content | Mail is a spam) * p(Spammed Mails)/ p(Content)

            """

            spam_size = labels.count(1)
            not_spam_size = labels.count(0)

            # Discrete Probabilities
            pSpam = spam_size / len(labels)
            pNotSpam = not_spam_size / len(labels)

            # Conditional Probabilities : p(words='a word'| class= 'spam') and p(words='a word'| class= 'normal')
            for i in range(len(train_data)):
                if train_labels[i] == 1:
                    for j in range(len(train_data[i][0])):
                        spam_vocabularies[train_data[i][0][j]] = spam_vocabularies[train_data[i][0][j]] + 1
                elif train_labels == 0:
                    for k in range(len(train_data[i][0])):
                        not_spam_vocabularies[train_data[i][0][k]] = not_spam_vocabularies[train_data[i][0][k]] + 1

            # loop in the test data
            test_predictions = []
            for i in range(len(test_data)):
                howLikelyisSpam = pSpam * classifier_utils.conditional_probability_of(test_data[i][0], 1,
                                                                                      classifier_utils.word_likelihood(
                                                                                          spam_vocabularies,
                                                                                          0, spam_size))
                howLikelyisNotSpam = pNotSpam * classifier_utils.conditional_probability_of(test_data[i][0], 0,
                                                                                            classifier_utils.word_likelihood(
                                                                                                not_spam_vocabularies,
                                                                                                0.0004, not_spam_size))
                if howLikelyisSpam > howLikelyisNotSpam:
                    test_predictions.append(1)
                else:
                    test_predictions.append(0)

            _, accuracy = confusion_matrix.accuracy(test_predictions, test_labels)
            _, sensitivity = confusion_matrix.sensivity(test_predictions, test_labels)
            _, specifity = confusion_matrix.specifity(test_predictions, test_labels)
            efficiency = confusion_matrix.efficiency(sensitivity, specifity)

            print("|========================================================|")
            print("| Model suceess for the", (e+1), ". iteration:                   |")
            print("|                                                        |")
            print("| The accuracy is: ", accuracy, "                     *")
            print("| The sensitivity is: ", sensitivity, "               *")
            print("| The specifity is: ", specifity, "                   *")
            print("| The efficiency is: ", efficiency, "                 *")
            print("|========================================================|")

            model_accuracies.append(accuracy)

        print("|=======================================================|")
        print("| Network challenged itself 5 times                     |")
        print("| In the mean the accuracy is: ", np.mean(model_accuracies), "          |")
        print("|=======================================================|")


if __name__ == '__main__':

    main()



