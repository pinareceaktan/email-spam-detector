import csv
import random
from subprocess import call
import argparse
import numpy as np

import refinement_lib
import numeric_lib
import feature_ext_lib
import classifier_utils
import confusion_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--use_sckit_learn', type=bool, default=True,
                    help='Use sckitlearn.normalize.scale method')
parser.add_argument('--no_of_runs', type=int, default=5,
                    help="How many times network should run")


def main(argv=None):

    with open('data.csv', newline='') as csv_file:
        spamreader = csv.reader(csv_file, delimiter=';')
        sentences = {}
        for row in spamreader:
            sentences[row[0]] = row[1]

        # Train and test data partition
        train_size = int(len(sentences) * .7)

        # Arrange train and test data
        sentences_list = []
        for sentence, value in sentences.items():
            sentences_list.append([sentence, value])

        # =========== Whole stuff here: Runs 5 times ===========================

        network_accuracy = []

        for e in range(5):

            random.shuffle(sentences_list)

            # These are raw data, not features, it is now required to extract features
            train_data = sentences_list[:train_size]
            #   test_data = sentences_list[train_size:]

            # Create vocabulary for each class, use train data
            spam_vocab_dict = dict()
            not_spam_vocab_dict = dict()

            for sentence in train_data:
                words = refinement_lib.make_all_lower(refinement_lib.tokenization(refinement_lib.get_words(sentence[0])))
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
                words = refinement_lib.make_all_lower(refinement_lib.tokenization(refinement_lib.get_words(sentence)))
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
                feature_sub_set.append(len(refinement_lib.get_words(sentence[0])))
                # 2. Punctuation count in text
                feature_sub_set.append(len(feature_ext_lib.get_punctuations(sentence[0])))
                # 3. Digit count in the text
                feature_sub_set.append(len(feature_ext_lib.get_digits(sentence[0])))
                # 4. Stop word counts in the text
                feature_sub_set.append(len(feature_ext_lib.get_stop_words(refinement_lib.get_words(sentence[0]))))
                # 5. Count of 2 lettered words in the text
                feature_sub_set.append(len(feature_ext_lib.get_short_words(sentence[0])))
                # 6. Count of words that include capitals in the text
                feature_sub_set.append(len(feature_ext_lib.get_capitals(sentence[0])))
                # 7. Average word len in text
                feature_sub_set.append(feature_ext_lib.get_avg_word_length(refinement_lib.get_words(sentence[0])))
                # 8. Count of white spaces in text
                feature_sub_set.append(feature_ext_lib.count_spaces(sentence[0]))
                # 9. Create sparse vectors of spam indicator presences sized 45
                feature_sub_set.extend(refinement_lib.vectorize(sentence[0], spam_indicators))
                # 10. Create sparse vectors of not spam indicator presences sized 45
                feature_sub_set.extend(refinement_lib.vectorize(sentence[0], not_spam_indicators))

                features.append(feature_sub_set)

                labels.append(1 if sentence[1] == 'spam' else 0)

            # Now get train features and labels
            train_features = features[0:train_size]
            train_labels = labels[0:train_size]
            test_features = features[train_size:]
            test_labels = labels[train_size:]

            # Normalization stuff here
            if FLAGS.use_sckit_learn:
                train_features = preprocessing.scale(train_features)
                test_features = preprocessing.scale(test_features)
                train_labels = numeric_lib.add_dimension(np.array(train_labels))
                test_labels = np.array(test_labels)
            else:
                train_features = np.array([numeric_lib.normalization(feature) for feature in train_features])
                test_features = np.array([numeric_lib.normalization(feature) for feature in test_features])
                train_labels = numeric_lib.add_dimension(np.array(train_labels))
                test_labels = np.array(test_labels)

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
            weight_0 = 2 * np.random.random((dim1, dim2)) - 1  # 98x8
            weight_1 = 2 * np.random.random((dim2, 1)) - 1  # 8x1

            # Training loop here:
            err_log = []
            go_on_task = True
            epochs = 0

            while go_on_task:
                # Design an early stopping mechanism, we do not want an overfit!
                # Use std on error logs, if std is smaller between batches
                # Than the network simply started to memorize and not learning a thing
                if epochs % 40 == 0 and epochs != 0:
                    improvement = np.std((err_log[epochs - 40:epochs - 20])) - np.std((err_log[epochs - 20:epochs]))

                    if abs(improvement) > 5e-10:
                        go_on_task = True
                    else:
                        go_on_task = False

                layer_0 = train_features
                layer_1 = classifier_utils.sigmoid(np.dot(layer_0, weight_0))
                layer_1 = np.array(layer_1)
                layer_2 = classifier_utils.sigmoid(np.dot(layer_1, weight_1))

                layer_2_err = train_labels - layer_2

                # perform back propagation
                layer_2_delta = layer_2_err * classifier_utils.derivative(layer_2)
                layer_1_error = layer_2_delta.dot(weight_1.T)
                layer_1_delta = layer_1_error * classifier_utils.derivative(layer_1)

                # update the weight vectors
                weight_1 += layer_1.T.dot(layer_2_delta)
                weight_0 += layer_0.T.dot(layer_1_delta)

                err_log.append(np.mean(layer_2_err))
                epochs = epochs + 1
                # Log error every 5 epoch
                if epochs % 5 == 0:
                    print(epochs, ".epoch, mse is=> ", classifier_utils.mse(layer_2_err))

            # evaluation on the testing data
            final_layer = classifier_utils.feed_forward(test_features, weight_0, weight_1)
            test_predictions = classifier_utils.threshold_network(final_layer)
            correct, score = confusion_matrix.accuracy(test_predictions, test_labels)

            print("Running the", (e+1), ".th", "test case")

            # printing the output
            print("total = ", len(final_layer))
            print("correct = ", correct)
            print("accuracy = ", score)

            network_accuracy.append(score)

            _, accuracy = confusion_matrix.accuracy(test_predictions, test_labels)
            _, sensitivity = confusion_matrix.sensivity(test_predictions, test_labels)
            _, specifity = confusion_matrix.specifity(test_predictions, test_labels)
            efficiency = confusion_matrix.efficiency(sensitivity, specifity)

            print("|========================================================|")
            print("| Model suceess for the", (e + 1), ". iteration:                   |")
            print("|                                                        |")
            print("| The accuracy is: ", accuracy, "                     *")
            print("| The sensitivity is: ", sensitivity, "               *")
            print("| The specifity is: ", specifity, "                   *")
            print("| The efficiency is: ", efficiency, "                 *")
            print("|========================================================|")

        print("|=======================================================|")
        print("| Network challenged itself 5 times                     |")
        for it in range(len(network_accuracy)):
            print("| ", it, "st iteration ended up with", network_accuracy[it], "      *")
        print("| In the mean the accuracy is: ", np.mean(network_accuracy),        "          |")
        print("|=======================================================|")


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


