import gym_maze
import gym
import numpy as np
import pickle
import math
import random
import os
from tqdm import trange

class Agent(object):
    def __init__(self, run_dir, save_dir):
        self.run_dir = run_dir
        self.save_dir = save_dir
        self.performance_log_filename = save_dir + 'data/performance_log.csv'
        self.pkl_filename = save_dir + 'data/agent.pkl'

        self.env = gym.make('maze-sample-3x3-v0')
        self.dim = 3
        self.os_n = self.dim * self.dim
        self.as_n = self.env.action_space.n

        self.c = 5
        # self.delta = .01
        self.delta = .1
        # self.epsilon = .01
        self.epsilon = .1
        self.discount_rate = .99


        if os.path.isfile(self.pkl_filename):
            print('loaded')
            saved_data = pickle.load(open(self.pkl_filename, 'rb'))
            self.Q_up = saved_data[0]
            self.Q_down = saved_data[1]
            self.n = saved_data[2]
            self.U = saved_data[3]
            self.iteration = saved_data[4]
        else:
            initial_value = math.log(1. * self.c * self.os_n * self.as_n / self.delta)
            self.Q_up = np.zeros([self.os_n, self.as_n]) + initial_value
            self.Q_down = np.zeros([self.os_n, self.as_n]) - initial_value
            self.n = np.ones([self.os_n, self.as_n])
            self.U = [list([j for j in range(self.as_n)]) for i in range(self.os_n)]
            self.iteration = 0



        self.model_free_action_elimination()


    def model_free_action_elimination(self):
        s = self.reset()
        while True:
            self.U[s] = list()
            V_down = np.max(self.Q_down[s])

            for a in range(self.as_n):
                if self.Q_up[s][a] >= V_down:
                    self.U[s].append(a)

            a = random.sample(self.U[s], 1)[0]
            s_next, r, done, _ = self.step(a)

            n = float(self.n[s][a])
            # print(self.Q_up[s][a], self.Q_down[s][a])
            
            # Update.
            self.Q_up[s][a] = (1. - 1./n) * self.Q_up[s][a] + 1./n * (r + self.discount_rate * self.V_up(s_next) + self.beta(n))
            self.Q_down[s][a] = (1. - 1./n) * self.Q_down[s][a] + 1./n * (r + self.discount_rate * self.V_down(s_next) - self.beta(n))
            self.n[s][a] += 1

            s = s_next
            # print(self.Q_up[s][a], self.Q_down[s][a])

            self.iteration += 1

            if self.iteration % 1000 == 0:
                print(self.iteration)
                self.check_stopping_conditions()
                self.save()    
            

            if done:
                # print('done! reward: %d, state: %d' % (r, s))
                state = self.reset()

    def step(self, a):
        s, r, done, _ = self.env.step(a)
        # r = max(r, 0.)
        return int(self.dim * s[0] +  s[1]), r, done, _

    def reset(self):
        s = self.env.reset()
        return int(self.dim * s[0] + s[1])
                
    def check_stopping_conditions(self):
        for s in range(self.os_n):
            for a in self.U[s]:
                if math.fabs(self.Q_up[s][a] - self.Q_down[s][a]) >= self.epsilon * (1-self.discount_rate) / 2.:
                    print('%f >= %f' % ((math.fabs(self.Q_up[s][a] - self.Q_down[s][a]), self.epsilon * (1-self.discount_rate) / 2)))
                    # print(np.absolute(self.Q_up - self.Q_down),np.mean(np.absolute(self.Q_up - self.Q_down)))
                    # exit()
                    return

        print('done')
        exit(0)

    def beta(self, k):
        return math.sqrt(math.log(1. * self.c * k * k * self.os_n * self.as_n / self.delta) / (1. * k))

    def V_up(self, s):
        return np.max(self.Q_up[s])

    def V_down(self, s):
        return np.max(self.Q_down[s])
            

    def save(self):
        pickle.dump((self.Q_up, self.Q_down, self.n, self.U, self.iteration), open(self.pkl_filename,'wb'))
    