"""
ADP
self-learnings
"""

import os
import pickle
import random
import time
# from CriticalNetwork import CriticNetwork, ActionNetwork
from ADP_linear import *
from m_nn import *

# 保存结果
WORK_FOLDER = r"D:\lesson\3-\人工智能\Final Project\Final\try"
CRITIC_NETWORK_SAVEPATH = WORK_FOLDER + r'\critic_network'

ME = 0
OPPONENT = 1
OBJECTIVE_MY = 1
OBJECTIVE_OPPONENTS = 0
MAX_env = 20

input_size = 4*7+2
hidden = 16
classes = 1

cur_env = np.array([[0 for i in range(MAX_env)] for j in range(MAX_env)])

env = ENV(cur_env, 20, 20, ME)

nn_m = NN(input_size, hidden, classes)
nn_o = NN(input_size, hidden, classes)
# 输入层大小要显式地算出来

if os.path.exists(CRITIC_NETWORK_SAVEPATH):
    nn_m.para = pickle.load(open(CRITIC_NETWORK_SAVEPATH, 'rb'))


def get_candidate(role):
    moves = env.get_actions(role)  # 返回临近的点
    if role == ME:
        move = nn_m.choose_best(moves, env)
    else:
        move = nn_o.choose_best(moves, env)
    return move


win_record = []
win_me = 0
win_rate = []
error = []

for i in range(5000):
    start_t = time.time()
    
    err = 0
    while not env.is_end():
        if env.role is None:
            env.role = random.choice([ME, OPPONENT])

        if env.role == ME:
            # print("my_move:", end=' ')
            move = get_candidate(env.role)
            err = nn_m.update(move, env)
        else:
            # print('OPPONENT\'S TURN:')
            # print("oppo_move:", end=' ')
            move = get_candidate(OPPONENT)
            env.make_move(move)
        if i % 100 == 0:
            nn_m.lr = 0.8*nn_m.lr
        if i % 10 == 0:
            nn_o.para = nn_m.para
    end_t = time.time()
    flag = env.is_end()
    if flag !=3:
        print('Game %s set. Winner: ' % i + ('ME' if flag == 1 else 'OPPONENT'))
    else:
        print("draw")
    print('train time:', end_t-start_t)
    error.append(err)
    win_record.append('ME' if flag == 1 else 'OPPONENT')
    if flag == 1:
        win_me += 1

    win_rate.append(win_me/(i+1))
    try:
        with open(CRITIC_NETWORK_SAVEPATH, 'wb') as f:
            pickle.dump(nn_m.para, f)
            f.close()
    except:
        print('Save model failed')
    env.reset(role = None)

import matplotlib.pyplot as plt
plt.plot(win_rate, color = 'orange')
plt.xlabel("Simulating times")
plt.ylabel("Winning rate")
plt.title("Winning rate during the process")
plt.savefig("res.png")
