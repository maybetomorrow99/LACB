from neuralucb_data import Bandit_multi
from neuralucb import NeuralUCBDiag
import numpy as np
import argparse
import pickle
import os
import time
import torch
from tqdm import tqdm

if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    parser = argparse.ArgumentParser(description='NeuralUCB')

    parser.add_argument('--size', default=15000, type=int, help='bandit size')
    parser.add_argument('--dataset', default='mnist', metavar='DATASET')
    parser.add_argument('--shuffle', type=bool, default=0, metavar='1 / 0', help='shuffle the data set or not')
    parser.add_argument('--seed', type=int, default=0, help='random seed for shuffle, 0 for None')
    parser.add_argument('--nu', type=float, default=1, metavar='v', help='nu for control variance')
    parser.add_argument('--lamdba', type=float, default=0.001, metavar='l', help='lambda for regularization')
    parser.add_argument('--hidden', type=int, default=100, help='network hidden size')

    # config
    batch_size = 16
    model_base_path = "./model/"
    model_name = "neural_ucb.pkl"
    model_path = model_base_path + model_name

    args = parser.parse_args()
    use_seed = None if args.seed == 0 else args.seed
    b = Bandit_multi(args.dataset, is_shuffle=args.shuffle, seed=use_seed)
    bandit_info = '{}'.format(args.dataset)

    l = NeuralUCBDiag(b.dim, args.lamdba, args.nu, args.hidden, load=False, load_path=model_path)
    ucb_info = '_{:.3e}_{:.3e}_{}'.format(args.lamdba, args.nu, args.hidden)

    regrets = []
    summ = 0

    batch = []
    batch_cnt = 0
    loss = 0
    for t in tqdm(range(b.size)):
        context, rwd = b.step()
        arm_select, nrm, sig, ave_rwd = l.select(context)
        r = rwd[arm_select]
        reg = np.max(rwd) - r
        summ += reg

        batch_cnt += 1
        if batch_cnt == batch_size:
            # calculate loss
            loss = l.train(context[arm_select], r)
            batch_cnt = 0
        else:
            l.add_batch(context[arm_select], r)

        # if t < 2000:
        #     loss = l.train(context[arm_select], r)
        # else:
        #     if t % 100 == 0:
        #         loss = l.train(context[arm_select], r)
        regrets.append(summ)
        if t % 500 == 0:
            # print('{}: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(t, summ, loss, nrm, sig, ave_rwd))
            print('Epoch: %d |Iter %d |reg: %.4f |loss: %.4f' % (1, t, summ, loss))

    model_name = '{}_{}.pkl'.format('neural_ucb', time.strftime("%m_%d_%H_%M", time.localtime()))
    # l.save_model(model_base_path + model_name)
    print('END')
