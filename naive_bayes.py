from numpy.random import seed
import pandas as pd
import numpy as np
from math import sqrt, exp, pi

# naive bayes =
# p(a|b) = p(b|a)p(a)/p(b)
''' Step 1: Separate by Class '''


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated.keys()):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


''' Step 2: Summarize Dataset '''


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg)**2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column))
                 for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


''' Step 3: Summarize data by class '''


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


''' Step 4: gaussian probability density function'''


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


''' Step 5: class probabilities '''


# P(class|data) = P(X|class) * P(class)
# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(
            total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(
                row[i], mean, stdev)
    return probabilities


# manage the calculation of the probabilities of a new row belonging to each class
# and selecting the class with the largest probability value.
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# manage the application of the Naive Bayes algorithm,
# first learning the statistics from a training dataset
# and using them to make predictions for a test dataset.
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    return predictions


def getmetrics(tp, tn, fp, fn):
    # accuracy, recall, precision
    return (tp + tn) / (tp + tn + fp + fn), tp / (tp + fn), tp / (tp + fp)


if __name__ == '__main__':
    seed(1)
    train = pd.read_csv("train_5feats.csv").to_numpy()
    test = pd.read_csv("test_5feats.csv")
    Test = test.to_numpy()
    predictions = naive_bayes(train, Test)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, actual in enumerate(test['Malicious']):
        if actual:
            if actual == predictions[i]:
                tp += 1
            else:
                fn += 1
        elif not actual:
            if actual == predictions[i]:
                tn += 1
            else:
                fp += 1
    accuracy, recall, precision = getmetrics(tp, tn, fp, fn)
    print(
        f"Accuracy: {round(accuracy,4)*100}%\nRecall: {round(recall,4)*100}%\nPrecision: {round(precision,4)*100}%"
    )
