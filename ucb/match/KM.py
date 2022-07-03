import numpy as np

zero_threshold = 1e-6
INF = 0x3f3f3f3f


class Solver:
    def __init__(self, matrix):
        """
        :param matrix: |l| x |r|
        :return: self.match[|r|]-->[0..|l|-1](matched) or -1(unmatched)
        """
        self.matrix = matrix
        self.trans = False  # transpose the matrix
        self.weight = self.trans_matrix()
        self.l_length = self.weight.shape[0]
        self.r_length = self.weight.shape[1]
        self.ex_l = np.zeros([self.l_length])  # |left| < |right|
        self.ex_r = np.zeros([self.r_length])
        self.vis_l = np.zeros([self.l_length], dtype=bool)
        self.vis_r = np.zeros([self.r_length], dtype=bool)
        self.match = np.full([self.r_length], -1)  # assign left to right
        self.slack = np.zeros([self.r_length], dtype=float)
        self.utility = 0

    def trans_matrix(self):
        # |request| < |agent|
        if self.matrix.shape[0] <= self.matrix.shape[1]:
            return self.matrix
        # |request| >= |agent|
        else:
            self.trans = True
            return self.matrix.T

    def KM(self):

        for i in range(self.l_length):
            self.ex_l[i] = np.max(self.weight[i, :])

        for i in range(self.l_length):
            self.slack = np.full(self.slack.shape, INF, dtype=float)

            while True:
                self.vis_l = np.zeros([self.l_length], dtype=bool)
                self.vis_r = np.zeros([self.r_length], dtype=bool)

                if self.dfs(i):
                    break

                d = INF
                for j in range(self.r_length):
                    if not self.vis_r[j]:
                        d = min(d, self.slack[j])

                for j in range(self.l_length):
                    if self.vis_l[j]:
                        self.ex_l[j] -= d
                for j in range(self.r_length):
                    if self.vis_r[j]:
                        self.ex_r[j] += d
                    else:
                        self.slack[j] -= d
        res = 0
        for i in range(self.r_length):
            if self.match[i] != -1:
                res += self.weight[self.match[i]][i]
        self.utility = res
        return self.utility

    def dfs(self, l):
        self.vis_l[l] = True

        for r in range(self.r_length):
            if self.vis_r[r]:
                continue
            gap = self.ex_l[l] + self.ex_r[r] - self.weight[l][r]

            if abs(gap) < zero_threshold:
                self.vis_r[r] = True
                if self.match[r] == -1 or self.dfs(self.match[r]):
                    self.match[r] = l
                    return True
            else:
                self.slack[r] = min(self.slack[r], gap)

        return False


def find_max_match(matrix):
    """
    :param matrix: |request| x |agent|
    :return: assign_agent[|r|]-->[0..|a|-1](matched) or -1(unmatched)  !!!confusing!!!
    """
    s = Solver(matrix)
    s.trans_matrix()
    s.KM()
    # print(s.utility)

    # assign right to left if |assign_agent| == |request|
    assign_agent = s.match
    if not s.trans:
        trans_match = np.full([matrix.shape[0]], -1)
        for i in range(s.match.shape[0]):
            if s.match[i] != -1:
                trans_match[s.match[i]] = i
        assign_agent = trans_match
    # print(s.match)
    # print(assign_agent)
    return assign_agent, s.utility


if __name__ == "__main__":
    matrix = np.array([[3, 5, 5, 4, 1],
                       [2, 2, 0, 2, 2],
                       [2, 4, 4, 1, 0],
                       [0, 1, 1, 0, 0],
                       [1, 2, 1, 3, 3]])
    find_max_match(matrix)
