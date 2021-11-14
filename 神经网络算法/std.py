import numpy as np
# sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid 函数求导
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:

    # 初始化参数
    def __init__(self, X, y, lr):
        self.input_layer = X
        self.W1 = np.random.rand(self.input_layer.shape[1], 3) # 注意形状
        self.W2 = np.random.rand(3, 1)
        self.y = y
        self.lr = lr
        self.output_layer = np.zeros(self.y.shape)

    # 前向传播
    def forward(self):

        # 实现公式 2，3
        self.hidden_layer = sigmoid(np.dot(self.input_layer, self.W1))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.W2))

    # 反向传播
    def backward(self):

        # 实现公式 5
        d_W2 = np.dot(self.hidden_layer.T, (2 * (self.output_layer - self.y) *
                      sigmoid_derivative(np.dot(self.hidden_layer, self.W2))))

        # 实现公式 6
        d_W1 = np.dot(self.input_layer.T, (
               np.dot(2 * (self.output_layer - self.y) * sigmoid_derivative(
               np.dot(self.hidden_layer, self.W2)), self.W2.T) * sigmoid_derivative(
               np.dot(self.input_layer, self.W1))))

        # 参数更新，实现公式 7
        # 因上方是 output_layer - y，此处为 -= 以保证符号一致
        self.W1 -= self.lr * d_W1
        self.W2 -= self.lr * d_W2

from matplotlib import pyplot as plt

# 测试数据
X = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

y = np.array([[0], [1], [0], [1]])

nn = NeuralNetwork(X, y, lr=0.1) # 定义模型
loss_list = [] # 存放损失数值变化

for i in range(1000):
    nn.forward() # 前向传播
    nn.backward() # 反向传播
    loss = np.sum((y - nn.output_layer) ** 2) # 计算平方损失
    loss_list.append(loss)

print("final loss:", loss)
plt.plot(loss_list) # 绘制 loss 曲线变化图
