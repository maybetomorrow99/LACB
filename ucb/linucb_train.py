import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from linucb import LinUCB
from sklearn.utils import shuffle
import time

MAX_CAPACITY = 44
workload_index = 11
ratio_index = 12


def train(ucb, agent_set):
    score = 0.0
    impressions = 0.0
    n_lines = 0.

    articles = [i for i in range(1, MAX_CAPACITY)]

    # train
    for epoch in range(5):
        for row in tqdm(agent_set):
            n_lines += 1
            chosen = row[workload_index]
            reward = row[ratio_index]
            user_features = [row[:workload_index]]
            # print(user_features.shape, articles.shape)
            calculated = ucb.recommend('', user_features, articles)

            if calculated == chosen:
                ucb.update(1, reward, arm=chosen)
                score += reward
                impressions += 1
            else:
                ucb.update(-1, 0)
    print(ucb.Aa)

    # save model
    # test
    cnt = 0
    n_lines = 0
    capacity_pred = list()

    for row in tqdm(agent_set):
        n_lines += 1

        chosen = row[workload_index]
        reward = row[ratio_index]
        user_features = [row[:workload_index]]
        calculated = ucb.recommend('', user_features, articles)

        capacity_pred.append(calculated)

        cnt += (abs(calculated - chosen) <= 5)
        if n_lines < 100:
            print(chosen, calculated)
    pd.DataFrame({'pred': capacity_pred}).to_csv('./data/' + 'capacity_pred_' + city_name + '.csv')
    print("acc: %.5f" % (cnt / len(agent_set)))


def pred(agent_set):
    ucb = LinUCB()
    ucb.load_model('./model/lucb_45.pkl')
    articles = [i for i in range(1, MAX_CAPACITY)]

    # test
    cnt = 0
    n_lines = 0
    capacity_pred = list()

    for row in tqdm(agent_set):
        n_lines += 1

        chosen = row[workload_index]
        reward = row[ratio_index]
        user_features = [row[:workload_index]]
        calculated = ucb.recommend('', user_features, articles)

        capacity_pred.append(calculated)

        cnt += (abs(calculated - chosen) <= 5)
        if n_lines < 100:
            print(chosen, calculated)
    print("acc: %.5f" % (cnt / len(agent_set)))


def run_train(agent_set):
    articles = {}
    ucb = LinUCB()

    # arm feature
    for i in range(1, MAX_CAPACITY):
        articles[i] = []
    ucb.set_articles(articles)
    train(ucb, agent_set)
    ucb.save_model('./model/lucb_{}_{}.pkl'.format(city_name, MAX_CAPACITY))
    # ucb.load_model('./model/lucb.pkl')


if __name__ == '__main__':
    city_name = "CityA"
    agent_set = pd.read_csv('./data/' + 'agent_set_' + city_name + '_over500.csv')
    agent_set = shuffle(agent_set)
    run_train(agent_set.values)

    # for heatmap
    # pred(agent_set.values)
