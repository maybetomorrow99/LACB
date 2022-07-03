import numpy as np
from datetime import datetime


class Logger:
    def __init__(self, path):
        self.path = path
        self.log_flag = True
        self.output = ""

    def log(self, s):
        self.output += s + '\n'

    def save(self):
        with open(self.path, 'w') as fout:
            fout.write(self.output)
