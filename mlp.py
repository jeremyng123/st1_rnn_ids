import copy
import pandas as pd
import numpy as np


# sigmoid
def logistic(x):
    return 1.0 / (1 + np.exp(-x))


# derivation
def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


def getmetrics(tp, tn, fp, fn):
    # accuracy, recall, precision
    try:
        accuracy, recall, precision = (tp + tn) / (tp + tn + fp + fn), tp / (
            tp + fn), tp / (tp + fp)
        return accuracy, recall, precision
    except ZeroDivisionError:
        return 0, 0, 0


def appoint_score(actual, pred):
    if actual:
        if actual == pred:
            scores['tp'] += 1
        else:
            scores['fn'] += 1
    elif not actual:
        if actual == pred:
            scores['tn'] += 1
        else:
            scores['fp'] += 1


if __name__ == '__main__':
    train = pd.read_csv("train_5feats.csv")
    test = pd.read_csv("test_5feats.csv")

    train_out = train.Malicious
    train_data = train.drop('Malicious', axis=1).to_numpy()
    train_size = train_data.shape[0]
    test_out = test.Malicious
    test_data = test.drop('Malicious', axis=1).to_numpy()
    test_size = test_data.shape[0]
    val = copy.deepcopy(test)

    # model settings
    LR = 1
    I_dim = train_data.shape[1]
    H_dim = 2  # arbitrary

    nepoch = 10**2
    weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
    weights_HtoO = np.random.uniform(-1, 1, H_dim)

    pre_activation_H = np.zeros(H_dim)
    post_activation_H = np.zeros(H_dim)
    """
    Training
    """
    # feedforwarding process
    print("Beginning training...")
    for epoch in range(nepoch):
        for sample in range(train_size):
            for node in range(H_dim):
                pre_activation_H[node] = np.dot(train_data[sample, :],
                                                weights_ItoH[:, node])
                post_activation_H[node] = logistic(pre_activation_H[node])
            pre_activation_O = np.dot(post_activation_H, weights_HtoO)
            post_activation_O = logistic(pre_activation_O)

            FE = post_activation_O - train_out[sample]

            # Backpropagation
            for H_node in range(H_dim):
                S_error = FE * logistic_deriv(pre_activation_O)
                gradient_OtoH = S_error * post_activation_H[H_node]

                for I_node in range(I_dim):
                    input_value = train_data[sample, I_node]
                    gradient_HtoI = S_error * weights_HtoO[
                        H_node] * logistic_deriv(
                            pre_activation_H[H_node]) * input_value
                    weights_ItoH[I_node, H_node] -= LR * gradient_OtoH
        if epoch % 10 == 0:
            scores = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            rval = val.sample(n=100)
            rval_out = rval.Malicious
            rval_data = rval.drop('Malicious', axis=1).to_numpy()
            rval_size = rval_data.shape[0]
            print(val)
            for sample in range(rval_size):
                for node in range(H_dim):
                    pre_activation_H[node] = np.dot(rval_data[sample, :],
                                                    weights_ItoH[:, node])
                    post_activation_H[node] = logistic(pre_activation_H[node])
                pre_activation_O = np.dot(post_activation_H, weights_HtoO)
                post_activation_O = logistic(pre_activation_O)

                if post_activation_O >= 0.5:
                    output = 1
                else:
                    output = 0

                appoint_score(test_out[sample], output)
            print(f"Epoch done: {epoch}\n\t{round(epoch/nepoch,3)*100}% done")
            accuracy, recall, precision = getmetrics(scores['tp'],
                                                     scores['tn'],
                                                     scores['fp'],
                                                     scores['fn'])
            print(
                f"\tAccuracy: {round(accuracy,4)*100}%\n\tRecall: {round(recall,4)*100}%\n\tPrecision: {round(precision,4)*100}%"
            )
    """
    Validation
    """
    scores = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    print("Beginning validation...")
    for sample in range(test_size):
        for node in range(H_dim):
            pre_activation_H[node] = np.dot(test_data[sample, :],
                                            weights_ItoH[:, node])
            post_activation_H[node] = logistic(pre_activation_H[node])
        pre_activation_O = np.dot(post_activation_H, weights_HtoO)
        post_activation_O = logistic(pre_activation_O)

        if post_activation_O >= 0.5:
            output = 1
        else:
            output = 0

        appoint_score(test_out[sample], output)
    accuracy, recall, precision = getmetrics(scores['tp'], scores['tn'],
                                             scores['fp'], scores['fn'])
    print(
        f"Accuracy: {round(accuracy,4)*100}%\nRecall: {round(recall,4)*100}%\nPrecision: {round(precision,4)*100}%"
    )
