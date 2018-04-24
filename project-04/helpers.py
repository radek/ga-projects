import numpy as np
from sklearn.metrics import mean_squared_error, classification_report, precision_score, recall_score, confusion_matrix


def get_data_scores(keys, data, predict):
    if not len(keys) or len(data) != len(predict):
        raise ValueError('Wrong input')
    scores = dict()
    for i in keys:
        scores[i.__name__] = i(data, predict)
    return scores


def print_collection(collection):
    for i in collection:
        print('{}:\n{}\n'.format(i, collection[i]))

def print_square_root(data):
    print('mean square root: {}\n'.format(np.sqrt(data)))

def print_predict_scores(data, predict):
    data_scores = get_data_scores([
        mean_squared_error,
        precision_score,
        recall_score,
        classification_report], data, predict)
    print_collection(data_scores)
    print_square_root(data_scores['mean_squared_error'])

