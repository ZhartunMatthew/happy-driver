from time import time
from sklearn.ensemble import GradientBoostingClassifier as Model
import numpy as np


def load_data(path, train=True):
    with open(path) as input_file:
        output_data = []
        for line in input_file:
            values = line.split(',')
            # used to skip first line with column names
            if not values[0].isdigit():
                continue

            temp_list = [float(x) for x in values[2:]] + [float(values[1])] if train else [float(x) for x in values]
            output_data.append(temp_list)

    print('Input data info.')
    print('\tVariable amount: %d, training set size %d' % (len(output_data[0]) - 1, len(output_data)))
    return np.array(output_data)


def create_submission(labels, answers, file_name):
    with open('../../data/submission_%s.csv' % file_name, 'w+') as f:
        f.write('id,target\n')
        for label, answer in zip(labels, answers):
            f.write('%d,%f\n' % (label, answer))
    return


def calculate_gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y))])
    g = g[np.lexsort((g[:, 2], -1*g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(y) + 1) / 2.0
    return gs / len(y)


def norm_gini(predicted, expected):
    return calculate_gini(expected, predicted) / calculate_gini(expected, expected)


class Estimator(object):

    def __init__(self, model):
        self.model = model

    def fit(self, data):
        x = data[:, :-1]
        y = data[:,  -1]
        self.model.fit(x, y)

    def predict(self, data):
        return self.model.predict_proba(data[:, 1:])[:, 1]

    def name(self):
        return self.model.__str__().split('(')[0]


def main():
    start = time()
    train_data = load_data('../../data/train.csv')
    test_data = load_data('../../data/test.csv', False)
    print('\tLoad data took: %.3f sec.' % (time() - start))

    estimator = Estimator(Model())
    estimator.fit(train_data)
    test_ans = estimator.predict(train_data)

    # gini computing for training_set
    gini = norm_gini(test_ans, train_data[:, -1])
    print('Test data gini coefficient: %f' % gini)

    # test data submission
    submission_answers = estimator.predict(test_data)
    create_submission(test_data[:, 0], submission_answers, estimator.name())
    print('Submission took: %.3f sec.' % (time() - start))


if __name__ == '__main__':
    main()

