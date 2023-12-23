import numpy as np
import struct
import os
import time

def show_matrix(mat, name):
    #print(name + str(mat.shape) + ' mean %f, std %f' % (mat.mean(), mat.std()))
    pass

def show_time(time, name):
    #print(name + str(time))
    pass

class FullyConnectedLayer(object):  # 全连接层初始化
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        print('\tFully connected layer with input %d, output %d.' % (self.num_input, self.num_output))
    def init_param(self, std=0.01):   # 参数初始化
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
        show_matrix(self.weight, 'fc weight ')
        show_matrix(self.bias, 'fc bias ')
    def forward(self, input):   # 前向传播计算
        start_time = time.time()
        self.input = input
        # TODO：全连接层的前向传播，计算输出结果
        self.output = self.input.dot(self.weight) + self.bias
        return self.output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：全连接层的反向传播，计算参数梯度和本层损失
        # d_weight 是本层的参数梯度, d_bias 是本层的偏置梯度, bottom_diff 是本层的损失
        # 参数梯度的计算公式是X^T*delta_Y L, 偏置梯度的计算公式是全1向量*delta_Y L, 全1向量的长度是输入的样本数 (也就是X的行数)
        # bottom_diff的计算公式是delta_Y L * W^T
        self.d_weight = self.input.T.dot(top_diff)
        self.d_bias = np.ones([1, self.input.shape[0]]).dot(top_diff)
        bottom_diff = top_diff.dot(self.weight.T)
        return bottom_diff
    def get_gradient(self):
        return self.d_weight, self.d_bias
    def update_param(self, lr):  # 参数更新
        # TODO：对全连接层参数利用参数进行更新
        # 权重更新公式是 W = W - lr * delta_W L, 偏置更新公式是 b = b - lr * delta_b L
        self.weight -= lr * self.d_weight
        self.bias -= lr * self.d_bias
    def load_param(self, weight, bias):  # 参数加载
        assert self.weight.shape == weight.shape
        assert self.bias.shape == bias.shape
        self.weight = weight
        self.bias = bias
    def save_param(self): # 参数保存
        return self.weight, self.bias

class ReLULayer(object):
    def __init__(self):
        print('\tReLU layer.')
    def forward(self, input):  # 前向传播的计算
        self.input = input
        # TODO：ReLU层的前向传播，计算输出结果
        # ReLU的计算公式是 max(0, X(i)), 对每一元素单独计算
        output = np.maximum(0, self.input)
        return output
    def backward(self, top_diff):  # 反向传播的计算
        # TODO：ReLU层的反向传播，计算本层损失
        # bottom_diff 是本层的损失. 计算公式是 delta_Y L = delta_Y L * (X >= 0), 对每个元素 X(i) 单独计算
        # 通过获取 x(i) < 0 的位置, 将 bottom_diff 中对应位置的元素置为 0, 其他位置的元素不变
        bottom_diff = top_diff * (self.input >= 0)
        return bottom_diff

class SoftmaxLossLayer(object):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    def get_loss(self, label): # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        # bottom_diff 是本层的损失. 计算公式是 delta_X L = (\hat{Y} - Y) / p
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff

