from time import time
from sklearn.ensemble import GradientBoostingClassifier as Model
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


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


def gini(actual, predicted):
    elements = np.asarray(np.c_[actual, predicted, np.arange(len(actual))], dtype=np.float)
    elements = elements[np.lexsort((elements[:, 2], -1*elements[:, 1]))]
    total_losses = elements[:, 0].sum()
    gini_sum = elements[:, 0].cumsum().sum() / total_losses
    gini_sum -= (len(actual) + 1) / 2.
    return gini_sum / len(actual)


def gini_xgb(predicted, expected):
    expected = expected.get_label()
    return 'gini', gini(expected, predicted) / gini(expected, expected)


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


class XEstimator(object):

    def __init__(self):
        self.params = {
                       'eta': 0.02, 'max_depth': 4,
                       'subsample': 0.9, 'colsample_bytree': 0.9,
                       'objective': 'binary:logistic',
                       'eval_metric': 'auc', 'seed': 99,
                       'silent': True}

        self.model = None

    def fit(self, data):
        x = data[:, :-1]
        y = data[:,  -1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=99)
        self.model = xgb.train(self.params,
                               xgb.DMatrix(x, label=y),
                               5000,
                               [(xgb.DMatrix(x_train, y_train), 'train'), (xgb.DMatrix(x_test, y_test), 'valid')],
                               feval=gini_xgb,
                               maximize=True,
                               verbose_eval=10)

    def predict(self, data):
        return self.model.predict(xgb.DMatrix(data[:, 1:]))

    def name(self):
        return self.model.__str__().split('(')[0]


def run(train_data, test_data, estimator):
    start = time()
    estimator.fit(train_data)

    submission_answers = estimator.predict(test_data)
    create_submission(test_data[:, 0], submission_answers, 'submission')
    print('Submission took: %.3f sec.' % (time() - start))


def main():
    train_data = load_data('../../data/train.csv')
    test_data = load_data('../../data/test.csv', False)
    run(train_data, test_data, XEstimator())


if __name__ == '__main__':
    main()
