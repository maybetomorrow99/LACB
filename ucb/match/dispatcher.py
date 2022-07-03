import sys
import os

pwd = os.path.dirname(os.path.realpath(__file__))
project_path = pwd[:pwd.find('ICDE23')] + 'ICDE23'
sys.path.append(project_path)
sys.path.append(project_path + "/ucb")

from simulator import Simulator
import numpy as np
from KM import find_max_match
from ucb.linucb import LinUCB
from ucb.neuralucb import NeuralUCBDiag
import pickle


class Top1(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None

    def match(self, matrix):
        len_r, len_a = matrix.shape
        assign_request = np.zeros(len_r, dtype=int)
        for i in range(len_r):
            agent_index = np.argmax(matrix[i])
            assign_request[i] = agent_index
        return assign_request

    def __str__(self):
        return 'Top-1 Recommendation (top1)'


class TopK(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None

    def match(self, matrix):
        len_r, len_a = matrix.shape
        assign_request = np.zeros(len_r, dtype=int)
        for i in range(len_r):
            # kth element
            k = np.random.randint(1, 4)
            agent_index = np.argsort(matrix[i])[-k]
            assign_request[i] = agent_index
        return assign_request

    def __str__(self):
        return 'Top-K Recommendation (topk)'


class RandomRec(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar):
        super().__init__(city_name, max_capacity, agent_over_bar)

    def match(self, matrix):
        len_r, len_a = matrix.shape
        assign_request = np.zeros(len_r, dtype=int)
        for i in range(len_r):
            s = sum(matrix[i])
            r = np.random.random() * s
            s = 0.0
            for j in range(len_a):
                s += matrix[i, j]
                if s >= r:
                    assign_request[i] = j
                    break
        return assign_request

    def __str__(self):
        return 'Randomized Recommendation (RR)'


class KMAssign(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar):
        super().__init__(city_name, max_capacity, agent_over_bar)

    def match(self, matrix):
        len_r, len_a = matrix.shape
        assign_request, utility = find_max_match(matrix)
        return assign_request

    def __str__(self):
        return 'KM Assign (km)'


class EmpiricalGreedy(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None

    def match(self, matrix):
        len_r, len_a = matrix.shape
        assign_request = np.zeros(len_r, dtype=int)
        for i in range(len_r):
            while True:
                agent_index = np.argmax(matrix[i])
                if self.agent_workload_day[agent_index] + 1 <= self.capacity[agent_index]:
                    assign_request[i] = agent_index
                    break
                else:
                    matrix[i, agent_index] = 0
        return assign_request

    def update_capacity(self, agent_batch):
        self.capacity = [self.MAX_CAPACITY for i in range(len(agent_batch))]
        return

    def __str__(self):
        return 'Capacity Top1 (cTop1)'


class CTopK(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None

    def match(self, matrix):
        len_r, len_a = matrix.shape
        assign_request = np.zeros(len_r, dtype=int)
        for i in range(len_r):
            while True:
                # kth element
                k = np.random.randint(1, 4)
                agent_index = np.argsort(matrix[i])[-k]
                if self.agent_workload_day[agent_index] + 1 <= self.capacity[agent_index]:
                    assign_request[i] = agent_index
                    break
                else:
                    matrix[i, agent_index] = 0
        return assign_request

    def update_capacity(self, agent_batch):
        self.capacity = [self.MAX_CAPACITY for i in range(len(agent_batch))]
        return

    def __str__(self):
        return 'Capacity TopK (ctopk)'


class LUCBOnline(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None
        self.ucb = LinUCB()

        articles = {}
        for i in range(1, self.MAX_CAPACITY):
            articles[i] = []
        self.ucb.set_articles(articles)
        self.agent_batch_for_train = None

    def match(self, matrix):
        len_r, len_a = matrix.shape
        for j in range(len_a):
            if self.agent_workload_day[j] >= self.capacity[j]:
                matrix[:, j] = 0
        assign_request, utility = find_max_match(matrix)
        return assign_request

    def update_capacity(self, agent_batch):
        self.capacity = np.zeros(self.agent_num, dtype=int)
        agent_batch_v = agent_batch.values
        articles = [i for i in range(1, self.MAX_CAPACITY)]
        for j in range(self.agent_num):
            user_features = [agent_batch_v[j]]
            calculated = self.ucb.recommend('', user_features, articles)
            self.capacity[j] = calculated
        print(self.capacity[:100])
        self.agent_batch_for_train = agent_batch
        return

    def update_day_score(self):
        self.agent_workload += self.agent_workload_day
        self.agent_workload_day_list.append(self.agent_workload_day)
        for j in range(self.agent_num):
            self.agent_score_day[j] *= self.convert_dis[j][self.agent_workload_day[j]]
        self.agent_score += self.agent_score_day
        self.agent_score_day_list.append(self.agent_score_day)

        # train ucb
        print('training')
        agent_batch_v = self.agent_batch_for_train.values
        for j in range(self.agent_num):
            if self.agent_workload_day[j] == 0:
                continue
            arm = self.agent_workload_day[j]
            r = self.convert_dis[j][self.agent_workload_day[j]]
            self.ucb.update_online(1, r, arm, agent_batch_v[j])

    def __str__(self):
        return 'LinUCB Online (ALOnline)'


class LinUCBAssign(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar, path=None):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None
        self.ucb = LinUCB()
        self.ucb.load_model(path)

    def match(self, matrix):
        len_r, len_a = matrix.shape
        for j in range(len_a):
            if self.agent_workload_day[j] >= self.capacity[j]:
                matrix[:, j] = 0
        assign_request, utility = find_max_match(matrix)
        return assign_request

    def update_capacity(self, agent_batch):
        self.capacity = np.zeros(self.agent_num, dtype=int)
        agent_batch_v = agent_batch.values
        articles = [i for i in range(1, self.MAX_CAPACITY)]
        for j in range(self.agent_num):
            user_features = [agent_batch_v[j]]
            calculated = self.ucb.recommend('', user_features, articles)
            self.capacity[j] = calculated
        return

    def __str__(self):
        return 'Assignment with LinUCB (AL)'


class NUCBOnline(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar, path=None):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None

        # bandit data info
        self.max_arm = self.MAX_CAPACITY
        self.n_arm = self.max_arm
        self.act_dim = 11
        self.dim = self.act_dim * self.n_arm

        self.ucb = NeuralUCBDiag(dim=self.dim, lamdba=0.001, nu=1, hidden=100)

        self.agent_batch_for_train = None
        self.epoch = 0

    def match(self, matrix):
        len_r, len_a = matrix.shape
        for j in range(len_a):
            if self.agent_workload_day[j] >= self.capacity[j]:
                matrix[:, j] = 0
        assign_request, utility = find_max_match(matrix)
        return assign_request

    def update_capacity(self, agent_batch):
        self.agent_batch_for_train = agent_batch
        self.capacity = np.zeros(self.agent_num, dtype=int)
        agent_batch_v = agent_batch.values
        for j in range(self.agent_num):
            context = self.get_context(agent_batch_v[j])
            arm_select, nrm, sig, ave_rwd = self.ucb.select(context)
            self.capacity[j] = arm_select
        print(self.capacity[:100])
        return

    def update_day_score(self):
        self.agent_workload += self.agent_workload_day
        self.agent_workload_day_list.append(self.agent_workload_day)
        for j in range(self.agent_num):
            self.agent_score_day[j] *= self.convert_dis[j][self.agent_workload_day[j]]
        self.agent_score += self.agent_score_day
        self.agent_score_day_list.append(self.agent_score_day)

        print('training')

        # train ucb
        batch_size = 16
        batch_cnt = 0
        agent_batch_v = self.agent_batch_for_train.values
        for j in range(self.agent_num):
            if self.agent_workload_day[j] == 0:
                continue
            context = self.get_context(agent_batch_v[j])
            arm = self.agent_workload_day[j]
            r = self.convert_dis[j][self.agent_workload_day[j]]
            batch_cnt += 1
            if batch_cnt == batch_size:
                loss = self.ucb.train(context[arm], r)
                batch_cnt = 0
            else:
                self.ucb.add_batch(context[arm], r)

    def get_context(self, agent):
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                                  self.act_dim] = agent
        return X

    def __str__(self):
        return 'Online Neural UCB Assign(AN)'


class NeuralUCBAssign(Simulator):
    def __init__(self, city_name, max_capacity, agent_over_bar, path=None):
        super().__init__(city_name, max_capacity, agent_over_bar)
        self.capacity = None
        self.ucb = NeuralUCBDiag(dim=1, load=True, load_path=path)

        # bandit data info
        self.max_arm = self.MAX_CAPACITY
        self.n_arm = self.max_arm
        self.act_dim = 11
        self.dim = self.act_dim * self.n_arm

        self.agent_batch = None

        with open(project_path + '/ucb/data/values_rl.pkl', 'rb') as file:
            d = pickle.load(file)
        self.top_agent_set = d['top_agent_set']
        self.cap_aware_values = d['cap_aware_values']
        self.factors = d['factors']
        self.cur_time = 0
        self.max_time = 144

    def match(self, matrix):
        # VFGA
        len_r, len_a = matrix.shape
        for j in range(len_a):
            if self.agent_workload_day[j] >= self.capacity[j]:
                matrix[:, j] = 0
        for j in range(len_a):
            aid = self.agent_batch['aid'].values
            if aid[j] in self.top_agent_set:
                for i in range(len_r):
                    delta_t = self.max_time - self.cur_time
                    v0_i = self.cap_aware_values[self.cur_time, self.capacity[j] - self.agent_workload_day[j]]
                    v1_i = self.cap_aware_values[
                        self.cur_time + delta_t, self.capacity[j] - self.agent_workload_day[j] - 1]
                    factor = self.factors[delta_t]
                    matrix[i, j] = matrix[i, j] + factor * v1_i - v0_i
        assign_request, utility = find_max_match(matrix)
        return assign_request

    def update_capacity(self, agent_batch):
        self.agent_batch = agent_batch
        self.cur_time += 1
        self.capacity = np.zeros(self.agent_num, dtype=int)
        agent_batch_v = agent_batch.values
        for j in range(self.agent_num):
            context = self.get_context(agent_batch_v[j])
            arm_select, nrm, sig, ave_rwd = self.ucb.select(context)
            self.capacity[j] = arm_select
        print(self.capacity[:100])
        return

    def get_context(self, agent):
        X = np.zeros((self.n_arm, self.dim))
        for a in range(self.n_arm):
            X[a, a * self.act_dim:a * self.act_dim +
                                  self.act_dim] = agent
        return X

    def __str__(self):
        return 'Assignment with Context Bandit (LACB)'


class NeuralUCBOptAssign(NeuralUCBAssign):
    def match(self, matrix):
        len_r, len_a = matrix.shape
        for j in range(len_a):
            if self.agent_workload_day[j] >= self.capacity[j]:
                matrix[:, j] = 0
        for j in range(len_a):
            aid = self.agent_batch['aid'].values
            if aid[j] in self.top_agent_set:
                for i in range(len_r):
                    delta_t = self.max_time - self.cur_time
                    v0_i = self.cap_aware_values[self.cur_time, self.capacity[j] - self.agent_workload_day[j]]
                    v1_i = self.cap_aware_values[
                        self.cur_time + delta_t, self.capacity[j] - self.agent_workload_day[j] - 1]
                    factor = self.factors[delta_t]
                    matrix[i, j] = matrix[i, j] + factor * v1_i - v0_i
        candidate_idx = np.argpartition(matrix, kth=-len_r, axis=1)[:, -len_r:]
        candidate_idx = np.unique(candidate_idx)
        new_matrix = matrix[:, candidate_idx]

        assign_request, utility = find_max_match(new_matrix)
        assign_request = candidate_idx[assign_request]
        return assign_request

    def __str__(self):
        return 'Assignment with Context Bandit Opt(LACB-Opt)'
