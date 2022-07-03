from dispatcher import *
import numpy as np
import time
import argparse
from logger import Logger
from datetime import datetime, timedelta
import sys
import os

pwd = os.path.dirname(os.path.realpath(__file__))
project_path = pwd[:pwd.find('ICDE23')] + 'ICDE23'
sys.path.append(project_path)
sys.path.append(project_path + "/ucb")

np.random.seed(42)


def run(dummy_flag=False):
    print('Begin!', str(sim))
    if args.city == 'Chengdu':
        start_day = '2021-07-01'
    elif args.city == 'Hangzhou':
        start_day = '2021-06-08'
    else:
        start_day = '2021-08-01'
    end_day = (datetime.strptime(start_day, "%Y-%m-%d") + timedelta(days=args.day)).strftime("%Y-%m-%d")
    day = None
    agent_batch = None
    sim_start_time = time.time()

    if not sim.synthetic:
        for t in sim.request_slot.groups:
            request_batch = sim.request_slot.get_group(t)
            day_t = request_batch['day'].iloc[0]
            print(t, len(request_batch), day_t)

            if day != day_t:
                if day: sim.update_day_score()
                day = day_t
                if day == end_day: break
                agent_batch = sim.get_agent_by_day(day)
                sim.update_capacity(agent_batch)
                sim.reset_day()
                logger.log(day_t)
            matrix = sim.get_matrix(len(request_batch), agent_batch)
            new_matrix = matrix
            if matrix.shape[0] < matrix.shape[1]:
                new_matrix = np.vstack((matrix, np.zeros((matrix.shape[1] - matrix.shape[0], matrix.shape[1]))))
            begin_time = time.time()
            assign_request = sim.match(new_matrix)[:matrix.shape[0]]
            end_time = time.time()
            sim.match_time += end_time - begin_time

            for i in range(len(assign_request)):
                sim.agent_workload_day[assign_request[i]] += 1
                sim.agent_score_day[assign_request[i]] += matrix[i, assign_request[i]]
    else:
        for t in range(sim.slot_num):
            request_batch = np.zeros((sim.one_batch_request_num, 2))
            day_t = t // sim.one_day_slot_num + 1
            print(t, len(request_batch), day_t)

            if day != day_t:
                if day: sim.update_day_score()
                day = day_t
                agent_batch = sim.get_agent_by_day('2021-08-01')
                sim.update_capacity(agent_batch)
                sim.reset_day()
                logger.log(str(day_t))

            matrix = sim.get_matrix(len(request_batch), agent_batch)
            new_matrix = matrix
            if matrix.shape[0] < matrix.shape[1]:
                new_matrix = np.vstack((matrix, np.zeros((matrix.shape[1] - matrix.shape[0], matrix.shape[1]))))
            begin_time = time.time()
            assign_request = sim.match(new_matrix)[:matrix.shape[0]]
            end_time = time.time()
            sim.match_time += end_time - begin_time

            for i in range(len(assign_request)):
                sim.agent_workload_day[assign_request[i]] += 1
                sim.agent_score_day[assign_request[i]] += matrix[i, assign_request[i]]

    sim_end_time = time.time()
    utility_total = sum(sim.agent_score)
    sim.save_pkl(path=exp_pkl_path)

    logger.log('Execution time: %.4f' % (sim_end_time - sim_start_time))
    logger.log('Match time: %.4f' % sim.match_time)
    logger.log('Total utility: %.4f' % utility_total)
    sim.agent_workload.sort()
    logger.log('Agent workload: %s' % str(sim.agent_workload[-100:]))
    logger.log('[%s] %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(sim)))
    logger.save()
    print('End!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='topk')
    parser.add_argument('--cap', type=int, default=99, help='max capacity')
    parser.add_argument('--city', type=str, default='Qingdao')
    parser.add_argument('--over', type=int, default=500, help='control the number of brokers')
    parser.add_argument('--day', type=int, default=14, help='request day')
    parser.add_argument('--path', type=str, default='', help='model path')
    parser.add_argument('--dummy', dest='dummy', action='store_true', default=False)
    parser.add_argument('--syn', dest='syn', action='store_true', default=False)
    parser.add_argument('--bro', type=int, default=2000, help='Broker Num')
    parser.add_argument('--req', type=int, default=50000, help='Request Num')
    args = parser.parse_args()
    if args.exp == 'topk':
        sim = TopK(args.city, args.cap, args.over)
    elif args.exp == 'top1':
        sim = Top1(args.city, args.cap, args.over)
    elif args.exp == 'rr':
        sim = RandomRec(args.city, args.cap, args.over)
    elif args.exp == 'km':
        sim = KMAssign(args.city, args.cap, args.over)
    elif args.exp == 'ega':
        sim = EmpiricalGreedy(args.city, args.cap, args.over)
    elif args.exp == 'ctopk':
        sim = CTopK(args.city, args.cap, args.over)
    elif args.exp == 'lucbo':
        sim = LUCBOnline(args.city, args.cap, args.over)
    elif args.exp == 'lucb':
        # path = 'dir/lucb.pkl'
        sim = LinUCBAssign(args.city, args.cap, args.over, path=project_path + args.path)
    elif args.exp == 'an':
        sim = NUCBOnline(args.city, args.cap, args.over)
    elif args.exp == 'nucb':
        # path = 'dir/nucb.pkl'
        sim = NeuralUCBAssign(args.city, args.cap, args.over, path=project_path + args.path)
    elif args.exp == 'nucbop':
        sim = NeuralUCBOptAssign(args.city, args.cap, args.over, path=project_path + args.path)
    else:
        raise ValueError('Unknown exp {} to run'.format(args.exp))
    print(sim.agent_num)

    # set the result path
    dummy_str = ''
    if args.dummy:
        dummy_str = '_dummy'

    if args.syn:
        exp_result_dir = project_path + '/ucb/result/Syn/%s_%d_%d' % (args.bro, args.req, args.day) + dummy_str
        exp_result_name = '/%s_%d_%d_%d_%d' % (args.exp, args.cap, args.bro, args.req, args.day)
        sim.set_synthetic_param(args.bro, args.req, args.day)
    else:
        exp_result_dir = project_path + '/ucb/result/%s/%s_%d_%d' % (
            args.city, args.city, args.day, sim.agent_num) + dummy_str
        exp_result_name = '/%s_%d_%s_%d_%d' % (args.exp, args.cap, args.city, args.day, sim.agent_num)

    if not os.path.exists(exp_result_dir):
        os.makedirs(exp_result_dir)
    if not os.path.exists(exp_result_dir + '/pkl'):
        os.makedirs(exp_result_dir + '/pkl')

    exp_result_path = exp_result_dir + exp_result_name + '.txt'
    exp_pkl_path = exp_result_dir + '/pkl' + exp_result_name + '.pkl'

    logger = Logger(path=exp_result_path)
    logger.log('[%s] %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(sim)))
    logger.log(str(args))

    run(dummy_flag=args.dummy)
