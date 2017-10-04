from time import time
from sklearn.decomposition import PCA
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


def compute_sparse(data):
    result = np.zeros((len(data[0])), float)
    for row in data:
        result += np.asarray(np.array(row == -1), int)

    return np.array((result / len(data)) * 100, int)


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


def add_extra_columns_to_train(train_data, addition):
    for i in range(len(addition[0])):
        for j in range(len(addition[0])):
            temp = addition[:, i] * addition[:, j]
            train_data = np.insert(train_data, 0, temp, 1)

    return train_data


def add_extra_columns_to_test(test_data, addition):
    for i in range(len(addition[0])):
        for j in range(len(addition[0])):
            temp = addition[:, i] * addition[:, j]
            test_data = np.insert(test_data, 1, temp, 1)

    return test_data


def run(train_data, test_data, estimator):
    start = time()
    estimator.fit(train_data)
    test_ans = estimator.predict(train_data)

    # gini computing for training_set
    gini = norm_gini(test_ans, train_data[:, -1])
    print('Training data gini coefficient: %f' % gini)
    # 3 add -
    # 0 add -
    # 0 add (exp) -

    # test data submission
    submission_answers = estimator.predict(test_data)
    create_submission(test_data[:, 0], submission_answers, estimator.name() + str(time()))
    print('Submission took: %.3f sec.' % (time() - start))


def main(comp):
    train_data = load_data('../../data/train.csv')
    test_data = load_data('../../data/test.csv', False)

    pca_train = PCA(n_components=comp)
    pca_train.fit(train_data[:, :-1])
    imp_train_data = pca_train.transform(train_data[:, :-1])
    train_data = add_extra_columns_to_train(train_data, imp_train_data)

    pca_test = PCA(n_components=comp)
    pca_test.fit(test_data[:, 1:])
    imp_test_data = pca_test.transform(test_data[:, 1:])
    test_data = add_extra_columns_to_test(test_data, imp_test_data)

    run(train_data, test_data, Estimator(Model(verbose=True)))


if __name__ == '__main__':
    main(0)
    main(3)
    main(10)
