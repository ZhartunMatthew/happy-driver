from time import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def load_data(path, train=True):
    with open(path) as input_file:
        output_data = []
        names = []
        for line in input_file:
            values = line.split(',')
            if not values[0].isdigit():
                names = values[2:] + [values[1]] if train else values
                continue

            temp_list = [float(x) for x in values[2:]] + [float(values[1])] if train else [float(x) for x in values]
            output_data.append(temp_list)

    print('Input data info.')
    print('\tVariable amount: %d, training set size %d' % (len(output_data[0]) - 1, len(output_data)))
    return np.array(output_data), names


def create_submission(labels, answers, file_name):
    with open('../../data/submission_%s.csv' % file_name, 'w+') as f:
        f.write('id,target\n')
        for label, answer in zip(labels, answers):
            f.write('%d,%f\n' % (label, answer))
    return


def gini(actual, predicted):
    elements = np.asarray(np.c_[actual, predicted, np.arange(len(actual))], dtype=np.float)
    elements = elements[np.lexsort((elements[:, 2], -1*elements[:, 1]))]
    total_losses = elements[:, 0].sum()
    gini_sum = elements[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)


def compute_gini(predicted, expected):
    expected = expected.get_label()
    return 'gini', gini(expected, predicted) / gini(expected, expected)


class XGBEstimator(object):

    def __init__(self):
        self.params = {'eta': 0.02, 'max_depth': 4,
                       'subsample': 0.9, 'colsample_bytree': 0.9,
                       'objective': 'binary:logistic',
                       'eval_metric': 'auc', 'silent': True, 'seed': 99}

        self.model = None
        self.iterations = 0

    def fit(self, data, iterations=200):
        x = data[:, :-1]
        y = data[:,  -1]

        self.iterations = iterations

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=99)

        self.model = xgb.train(self.params, xgb.DMatrix(x, label=y), self.iterations,
                               [(xgb.DMatrix(x_train, y_train), 'train'), (xgb.DMatrix(x_test, y_test), 'test')],
                               feval=compute_gini, maximize=True, verbose_eval=10)

    def predict(self, data):
        return self.model.predict(xgb.DMatrix(data[:, 1:]))

    def name(self):
        return 'xgboost_%d' % self.iterations


def run(train_data, test_data, estimator):
    start = time()
    estimator.fit(train_data, 1000)

    submission_answers = estimator.predict(test_data)
    create_submission(test_data[:, 0], submission_answers, '_%s' % estimator.name())
    print('Submission took: %.3f sec.' % (time() - start))


def main():
    print('Loading')
    train_data, train_names = load_data('../../data/train.csv')
    test_data, test_names = load_data('../../data/test.csv', False)

    # deleting unnecessary calculated field (it will increase gini score)
    train_data = np.delete(train_data, [i for i in range(len(train_names)) if train_names[i].startswith('ps_calc_')], 1)
    test_data = np.delete(test_data, [i for i in range(len(test_names)) if test_names[i].startswith('ps_calc_')], 1)

    print('Predicting')
    run(train_data, test_data, XGBEstimator())


if __name__ == '__main__':
    main()
