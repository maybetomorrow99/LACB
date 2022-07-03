import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.optim as optim
import pickle


class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        return self.fc3(x)


class NeuralUCBDiag:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100, load=False, load_path=''):
        self.cuda_flag = torch.cuda.is_available()
        if not load:
            self.func = Network(dim, hidden_size=hidden)
            self.lamdba = lamdba
            self.total_param = sum(p.numel() for p in self.func.parameters() if p.requires_grad)
            self.U = lamdba * torch.ones((self.total_param,))
            self.nu = nu
        else:
            self.load_model(load_path)

        self.context_list = []
        self.reward = []

        # for heatmap
        self.arm_utility_list = []

        if self.cuda_flag:
            self.func = self.func.cuda()
            self.U = self.U.cuda()

    def select(self, context):
        tensor = torch.from_numpy(context).float()
        if self.cuda_flag:
            tensor = tensor.cuda()
        mu = self.func(tensor)
        g_list = []
        sampled = []
        ave_sigma = 0
        ave_rew = 0
        for fx in mu:
            self.func.zero_grad()
            fx.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.func.parameters()])  # multi process blocks here
            # print(g)
            g_list.append(g)
            sigma2 = self.lamdba * self.nu * g * g / self.U
            sigma = torch.sqrt(torch.sum(sigma2))

            sample_r = fx.item() + sigma.item()

            sampled.append(sample_r)
            ave_sigma += sigma.item()
            ave_rew += sample_r

        # for heatmap
        # self.arm_utility_list.append(sampled)
        # print(sampled)

        arm = np.argmax(sampled)
        self.U += g_list[arm] * g_list[arm]
        return arm, g_list[arm].norm().item(), ave_sigma, ave_rew

    def train(self, context, reward):
        # self.context_list.append(torch.from_numpy(context.reshape(1, -1)).float())
        self.context_list.append(context.reshape(1, -1))
        self.reward.append(reward)
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba)
        length = len(self.reward)
        index = np.arange(length)

        np.random.shuffle(index)
        cnt = 0
        tot_loss = 0

        # while True:
        batch_loss = 0.0
        # for idx in index:
        c = torch.from_numpy(np.array(self.context_list).reshape(length, -1)).float()
        r = torch.from_numpy(np.array(self.reward).reshape(-1, 1)).float()
        if self.cuda_flag:
            c = c.cuda()
            r = r.cuda()
        for child in self.func.children():
            cnt += 1
            if cnt < 3:
                for param in child.parameters():
                    param.requires_grad = False
        optimizer.zero_grad()
        delta = self.func(c) - r
        loss = sum(delta * delta)
        loss.backward()
        optimizer.step()
        batch_loss += loss.item()
        tot_loss += loss.item()

        # cnt += 1
        # if cnt >= 1000:
        #     return tot_loss / 1000
        # if batch_loss / length <= 1e-3:

        self.context_list = []
        self.reward = []
        return batch_loss / length

    def train_batch(self, batch):
        return

    def add_batch(self, context, reward):
        self.context_list.append(context.reshape(1, -1))
        self.reward.append(reward)
        return

    def save_model(self, path):
        d = {'lamdba': self.lamdba, 'U': self.U, 'nu': self.nu}
        with open(path, 'wb') as f:
            pickle.dump(d, f)

        # save nn
        path_nn = path.rstrip('.pkl') + '_nn.pkl'
        torch.save(self.func, path_nn)
        return

    def load_model(self, path):
        with open(path, 'rb') as file:
            d = pickle.load(file)
        self.lamdba = d['lamdba']
        self.U = d['U']
        self.nu = d['nu']

        # load nn 'dir/nucb_nn.pkl'
        path_nn = path.rstrip('.pkl') + '_nn.pkl'
        self.func = torch.load(path_nn)
        return
