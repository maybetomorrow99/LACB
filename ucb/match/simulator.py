import time
import pandas as pd
import numpy as np
import sys
import os
import pickle
from kesim import *

pwd = os.path.dirname(os.path.realpath(__file__))
project_path = pwd[:pwd.find('ICDE23')] + 'ICDE23'
sys.path.append(project_path)
sys.path.append(project_path + "/ucb")


class Simulator:
    def __init__(self, city_name, max_capacity, agent_over_bar, synthetic=False):
        self.city_name = city_name
        self.MAX_CAPACITY = max_capacity
        self.agent_over_bar = agent_over_bar
        self.synthetic = synthetic

        self.request_slot = self.read_request_data()
        self.agent_set = self.read_agent_data()
        self.convert_dis = ke_convert()
        self.agent_set_id = self.agent_set['aid'].unique()
        self.agent_set_id.sort()
        self.agent_num = len(self.agent_set_id)

        self.agent_score = np.zeros(self.agent_num, dtype=float)
        self.agent_workload = np.zeros(self.agent_num, dtype=int)
        self.agent_score_day = np.zeros(self.agent_num, dtype=float)
        self.agent_workload_day = np.zeros(self.agent_num, dtype=int)
        self.agent_score_day_list = list()
        self.agent_workload_day_list = list()
        self.match_time = 0
        self.match_time_day_list = list()

    def read_request_data(self):
        request_set = pd.read_csv(project_path + '/ucb/data/' + 'request_set_' + self.city_name + '.csv')
        print(request_set.shape)
        return request_set.groupby('slot')

    def read_agent_data(self):
        agent_set = pd.read_csv(
            project_path + '/ucb/data/' + 'agent_set_' + self.city_name + '_over' + str(self.agent_over_bar) + '.csv')
        return agent_set

    def set_synthetic_param(self, agent_num, request_num, day_num):

        self.synthetic = True
        self.agent_num = agent_num
        self.request_num = request_num
        self.day_num = day_num
        self.agent_score = np.zeros(self.agent_num, dtype=float)
        self.agent_workload = np.zeros(self.agent_num, dtype=int)
        self.agent_score_day = np.zeros(self.agent_num, dtype=float)
        self.agent_workload_day = np.zeros(self.agent_num, dtype=int)
        self.one_day_slot_num = 144
        self.one_batch_request_num = int((self.request_num / day_num) // self.one_day_slot_num)  # 这里可能会亏单，不过是大家一起亏
        self.slot_num = self.one_day_slot_num * day_num + 2

        return

    def get_agent_by_day(self, day):
        agent_day = self.agent_set[self.agent_set['day'] == day]
        agent_set_batch = pd.DataFrame(columns=['aid'], data=self.agent_set_id)
        result = pd.merge(agent_set_batch, agent_day, how='left', on='aid')
        nan_index = result[result['quality_score_v3'].isnull()].index
        for i in nan_index:
            result.loc[i] = self.agent_set[self.agent_set['aid'] == result.loc[i]['aid']].sample().values[0]
        if self.synthetic:
            result = result[:self.agent_num]
        return result

    def get_weight(self, agent):
        score = ke_get_weight(agent)
        return score

    def get_matrix(self, request_num, agent_batch):
        weight_matrix = np.zeros((request_num, len(agent_batch)))
        agent_batch = agent_batch.values
        for i in range(request_num):
            for j in range(len(agent_batch)):
                weight_matrix[i, j] = self.get_weight(agent_batch[j])
        return weight_matrix

    def update_capacity(self, *args):
        pass

    def update_day_score(self):
        self.agent_workload += self.agent_workload_day
        self.agent_workload_day_list.append(self.agent_workload_day)
        for j in range(self.agent_num):
            self.agent_score_day[j] *= self.convert_dis[j][self.agent_workload_day[j]]
        self.agent_score += self.agent_score_day
        self.agent_score_day_list.append(self.agent_score_day)

    def reset_day(self):
        self.agent_score_day = np.zeros(self.agent_num, dtype=float)
        self.agent_workload_day = np.zeros(self.agent_num, dtype=int)

    def save_pkl(self, path):
        # use pickle
        d = {'agent_score': self.agent_score, 'agent_workload': self.agent_workload,
             'agent_score_list': self.agent_score_day_list, 'agent_workload_list': self.agent_workload_day_list}
        with open(path, 'wb') as f:
            pickle.dump(d, f)
        return
