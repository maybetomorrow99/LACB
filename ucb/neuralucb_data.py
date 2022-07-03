from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd

MAX_CAPACITY = 45
CITY_NAME = 'CityA'
AGENT_OVER_BAR = 500


class Bandit_multi:
    def __init__(self, name, is_shuffle=True, seed=None):
        # Fetch data
        """
        if name == 'mnist':
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'covertype':
            X, y = fetch_openml('covertype', version=3, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'MagicTelescope':
            X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        elif name == 'shuttle':
            X, y = fetch_openml('shuttle', version=1, return_X_y=True)
            # avoid nan, set nan as -1
            X[np.isnan(X)] = - 1
            X = normalize(X)
        else:
            raise RuntimeError('Dataset does not exist')
        """
        self.city_name = CITY_NAME
        df_data = pd.read_csv('./data/' + 'agent_set_' + self.city_name + '_over' + str(AGENT_OVER_BAR) + '.csv')

        X = df_data.values
        y = df_data['workload'].values.astype(int)
        R = df_data['convert_ratio'].values

        print(X.shape, y.shape, R.shape)

        self.max_capacity = MAX_CAPACITY
        # limit capacity
        y[y >= MAX_CAPACITY] = MAX_CAPACITY - 1

        # Shuffle data
        if is_shuffle:
            self.X, self.y = shuffle(X, y, R, random_state=seed)
        else:
            self.X, self.y, self.R = X, y, R

        # generate one_hot coding:
        # self.y_arm = OrdinalEncoder(dtype=np.int).fit_transform(self.y.reshape((-1, 1)))
        self.y_arm = self.y.reshape(-1, 1)
        # cursor and other variables
        self.cursor = 0
        self.size = self.y.shape[0]
        # self.n_arm = np.max(self.y_arm) + 1
        self.n_arm = MAX_CAPACITY
        self.dim = self.X.shape[1] * self.n_arm
        self.act_dim = self.X.shape[1]

    def step(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                                  self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        rwd = np.zeros((self.n_arm,))  # out of index
        # rwd[arm] = 1
        rwd[max(arm - 5, 0):arm] = self.R[self.cursor]
        rwd[arm:min(arm + 6, self.n_arm)] = self.R[self.cursor]
        self.cursor += 1
        return X, rwd

    def step_offline(self):
        assert self.cursor < self.size
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                                  self.act_dim] = self.X[self.cursor]
        arm = self.y_arm[self.cursor][0]
        r = self.R[self.cursor]
        self.cursor += 1
        return X, arm, r

    def finish(self):
        return self.cursor == self.size

    def reset(self):
        self.cursor = 0
