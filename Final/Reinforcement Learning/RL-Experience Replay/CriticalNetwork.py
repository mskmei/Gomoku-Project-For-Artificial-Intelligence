import random
import numpy as np
import os
import pickle


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

    def update(self, lr, mode=0):
        # 随机梯度下降进行更新
        self.w -= lr * self.weight_gradient
        self.b -= lr * self.bias_gradient



class CriticNetwork:
    def __init__(self, params):

        # 超参数
        self.ALPHA = 1
        self.LEARNING_RATE = 0.1
        self.MAGNIFY = 1
        activator = sigmoid()

        # 建构神经网络
        self.layers = []
        for i in range(len(params) - 1):
            self.layers.append(Layer(params[i], params[i + 1], activator))

    # 神经网络也可以从文件提取
    def load_layers(self, filepath):
        if os.path.exists(filepath):
            self.layers = pickle.load(open(filepath, 'rb'))
        else:
            raise Exception('File ' + filepath + ' does not exist')

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
        # 前向计算，返回获胜概率
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
    def calc_gradient(self, error):
        delta = -error
        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

    # 后向迭代
    def back_propagation(self, V,next_V, reward):

        learning_rate = self.LEARNING_RATE * self.MAGNIFY if next_V == 0 else self.LEARNING_RATE
        error = self.ALPHA * (reward + 0.99*next_V - V)
        self.calc_gradient(error)

        for layer in self.layers:
            layer.update(learning_rate, mode=0)
        return error


class ActionNetwork:
    def __init__(self, objective=1, EPSILON=0.0):
        self.objective = objective
        self.EPSILON = EPSILON

    def forward(self, actions, values):
        if random.random() < self.EPSILON:
            random_index = random.randint(0, len(actions) - 1)
            return actions[random_index], values[random_index]
        else:
            best_value=float("-inf")
            best_action=None
            for action, value in zip(actions, values):
                if value>best_value:
                    best_action, best_value = action, value
            return best_action, best_value
