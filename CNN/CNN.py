import numpy as np


class Conv3x3(object): # 卷积层
    # 一个3x3的filter

    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.ones((num_filters, 3, 3))

    def regions_list(self, image):
        L = []
        n, m = image.shape
        for i in range(n - 2):
            for j in range(m - 2):
                L.append((image[i:i+3, j:j+3], i, j))
        return L
    
    def foward(self, input):# 正向传播
        n, m = input.shape
        output = np.zeros((self.num_filters, n - 2, m - 2))
        for (image,i,j) in self.regions_list(input):
            for t in range(self.num_filters):
                output[t, i, j] = np.sum(image * self.filters[t])
        return output

class MaxPool2(object): # 池化层
    def regions_list(self, image):
        output = []
        num_filters, n, m = image.shape
        for t in range(num_filters):
            L = []
            for i in range(0, n, 2):
                for j in range(0, m, 2):
                    L.append((image[t][i:i+2, j:j+2], i//2, j//2))
            output.append(L)
        return output
    def foward(self, input):
        num_filters, n, m = input.shape
        output = np.zeros((num_filters, n//2, m//2))
        Regins_List = self.regions_list(input)
        for t in range(num_filters):
            for (image, i, j) in Regins_List[t]:
                output[t, i, j] = np.max(image)
        return output

class Sofrmax(object): #输出层
    def __init__(self, input_len, nodes_num):
        self.W = np.random.randn((input_len, nodes_num)) / input_len
        self.B = np.random.zeros(nodes_num)
    def foward(self, input):
        input = input.flatten()
        input_len, nodes_num = self.W.shape
        out = np.dot(input, self.W) + sel.B
        out_exp = np.exp(out)
        return out_exp / np.sum(out_exp, axis=0)

conv = Conv3x3(8)
pool = MaxPool2()
softmax = Sofrmax(13 * 13 * 8, 10)

#def Forward(image, label):
    #out = conv.foward((image / 255) - 0.5)

#B = np.array([[
#   [1, 2, 3, 4],
#   [5, 6, 7, 8],
#   [9, 10, 11, 12],
#   [13, 14, 15, 16]
#]])
#A = MaxPool2()
#print(A.foward(B))
