import random


import os
import pickle
import numpy as np

MAX = 1000
JUDGE = 1

class sigmoid(object):
    def forward(self, weight):
        return 1.0 / (1.0 + np.exp(-weight))

    def backward(self, output):
        return output * (1 - output)


class Layer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        input_size：输入维度
        output_size：输出维度
        activator：激活函数
        w:权重
        b:bias
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        
        self.w = np.random.uniform(-0.5, 0.5, (output_size, input_size))
        self.b = np.zeros((output_size, 1))
        self.gradient_w = np.zeros((output_size, input_size))
        self.gradient_b = np.zeros((output_size, 1))

    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = self.activator.forward(np.dot(self.w, self.input_data) + self.b)

    def backward(self, previous):
        # previous_为后一层传入的误差
        # output_former为传入前一层的误差
        self.product = previous * self.activator.backward(self.output_data)
        output_former = np.dot(self.w.T, self.product)
        self.weight_gradient = np.dot(self.product, self.input_data.T)
        self.bias_gradient = self.product
        self.gradient_w += self.weight_gradient
        self.gradient_b += self.bias_gradient
        return output_former

    def SGD(self, lr):
        # 随机梯度下降进行更新
        self.w -= lr * self.weight_gradient
        self.b -= lr * self.bias_gradient



class CriticNetwork:
    def __init__(self, parameters):

        # 超参数
        self.alp = 1
        self.LR = 0.1
        self.mag = 1
        activator = sigmoid()

        # 建构神经网络
        self.layers = []
        for i in range(len(parameters) - 1):
            self.layers.append(Layer(parameters[i], parameters[i + 1], activator))

    def load_layers(self, dire):
        if os.path.exists(dire):
            self.layers = pickle.load(open(dire, 'rb'))
        else:
            raise Exception('No such files')

    # 特征提取，此处的架构与ADP-MCTS一致
    def extract_features(self, env):
        # logDebug('env features: '+ str(env.features))
        role = env.role
        features = env.features(role)
        num = len(features)
        flattened_features = []
        for i in range(num//2):
            flattened_features.append(features[i])
            flattened_features.append(features[i+num//2])
            flattened_features.append(role)
            flattened_features.append(1-role)
        return np.asarray(flattened_features + [role, 1-role]).reshape(-1, 1)

        # 7*4 + 2 =30

    def forward(self, env):
        if env.is_end() == 1:
            return 1 
        elif env.is_end() == 2:
            return 0
        output = self.extract_features(env)

        for layer in self.layers:
            layer.forward(output)
            output = layer.output_data
        return float(output)

    # 计算梯度
    def grad_compute(self, error):
        temp = -self.alp * error  # 注意！！！
        for layer in self.layers[::-1]:
            temp = layer.backward(temp)
        # return error * temp
        return temp

    # 向后传播
    def back_propagation(self, env, move, role, reward):
        V = self.forward(env)
        env.make_move(move)
        next_V = self.forward(env)

        learning_rate = self.LR * self.mag if next_V == 0 else self.LR
        error = self.alp * (reward + next_V - V)
        self.grad_compute(error)

        for layer in self.layers:
            layer.SGD(learning_rate)
        return error


class Action_Choose:
    def __init__(self, objective=JUDGE, threshold=0.0):
        self.objective = objective
        self.threshold = threshold

    def forward(self, actions, values):
        if random.random() < self.threshold:  # threshold greedy
            action_value = list(
                filter(lambda x: abs(x[1] - self.objective) < 1, zip(actions, values))) 
            if len(action_value) > 0:
                rd_ind = random.randint(0, len(action_value) - 1)
                return action_value[rd_ind][0], action_value[rd_ind][1]
            else:
                rd_ind = random.randint(0, len(actions) - 1)
                return actions[rd_ind], values[rd_ind]
        else:
            max_dif = MAX-1
            best_value = None
            best_move = None

            for action, value in zip(actions, values):
                diff = abs(value - self.objective)
                if diff < max_dif:
                    max_dif, best_move, best_value = diff, action, value
            return best_move, best_value
