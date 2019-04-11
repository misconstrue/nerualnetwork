# 该神经网络采用8*4*1的结构（从第一层隐藏层开始到输出层）
import numpy as np
import matplotlib.pyplot as plot
train_data = []  # 训练数据列表
y_trues = []
test_data = []  # 测试数据列表
y_tests = []
for i in np.arange(0, 2*np.pi, 0.02):  # 得出训练样本点 315个
    train_data.append(i)
for i in train_data:  # 训练样本对应的真值
    y_trues.append(np.sin(i))
for i in np.arange(0, 2*np.pi, 0.005):  # 测试数据密度约为训练样本的四倍 1257
    test_data.append(i)
for i in test_data:  # 得出测试的真值 方便测试使用
    y_tests.append(np.sin(i))
dictionary = dict(zip(train_data, y_trues))  # 将训练数据转化为字典的数据结构 方便后面训练时候调用


def sigmoid(x):  # 非线性函数  此处采用 sigmoid函数
    return 1/(1+np.exp(-x))


class NeuralNetwork:  # 创建神经网络类
    def __init__(self):
        self.epoch = []  # 迭代次数
        self.loss_value = []  # 损失
        self.w11 = np.random.normal()  # 第一层权重  # 权重参数初始化，采用了随机初始化的方式，后面参数初始化类似。
        self.w12 = np.random.normal()
        self.w13 = np.random.normal()
        self.w14 = np.random.normal()
        self.w15 = np.random.normal()
        self.w16 = np.random.normal()
        self.w17 = np.random.normal()
        self.w18 = np.random.normal()
        self.w21 = np.random.normal()  # 第二层权重
        self.w22 = np.random.normal()
        self.w23 = np.random.normal()
        self.w24 = np.random.normal()
        self.w25 = np.random.normal()
        self.w26 = np.random.normal()
        self.w27 = np.random.normal()
        self.w28 = np.random.normal()
        self.w29 =  np.random.normal()
        self.w210 = np.random.normal()
        self.w211 = np.random.normal()
        self.w212 = np.random.normal()
        self.w213 = np.random.normal()
        self.w214 = np.random.normal()
        self.w215 = np.random.normal()
        self.w216 = np.random.normal()
        self.w217 = np.random.normal()
        self.w218 = np.random.normal()
        self.w219 = np.random.normal()
        self.w220 = np.random.normal()
        self.w221 = np.random.normal()
        self.w222 = np.random.normal()
        self.w223 = np.random.normal()
        self.w224 = np.random.normal()
        self.w225 = np.random.normal()
        self.w226 = np.random.normal()
        self.w227 = np.random.normal()
        self.w228 = np.random.normal()
        self.w229 = np.random.normal()
        self.w230 = np.random.normal()
        self.w231 = np.random.normal()
        self.w232 = np.random.normal()
        self.w31 = np.random.normal()  # 第三层权重
        self.w32 = np.random.normal()
        self.w33 = np.random.normal()
        self.w34 = np.random.normal()
        self.w35 = np.random.normal()
        self.w36 = np.random.normal()
        self.w37 = np.random.normal()
        self.w38 = np.random.normal()
        self.b1 = np.random.normal()  # 第一层偏差  # 偏差参数初始化
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()
        self.b5 = np.random.normal()
        self.b6 = np.random.normal()
        self.b7 = np.random.normal()
        self.b8 = np.random.normal()
        self.b9 = np.random.normal()  # 第二层偏差
        self.b10 = np.random.normal()
        self.b11 = np.random.normal()
        self.b12 = np.random.normal()
        self.b13 = np.random.normal()  # 第三层偏差

    def feedforward(self, x):  # 前向传播  测试的时候求用来求取结果
        f1 = sigmoid(self.w11 * x + self.b1)  # 第一层传播结果
        f2 = sigmoid(self.w12 * x + self.b2)
        f3 = sigmoid(self.w13 * x + self.b3)
        f4 = sigmoid(self.w14 * x + self.b4)
        f5 = sigmoid(self.w15 * x + self.b5)
        f6 = sigmoid(self.w16 * x + self.b6)
        f7 = sigmoid(self.w17 * x + self.b7)
        f8 = sigmoid(self.w18 * x + self.b8)
        f9 = sigmoid(self.w21 * f1 + self.w22 * f2 + self.w23 * f3 + self.w24 * f4 +  # 第二层传播结果
                     self.w25 * f5 + self.w26 * f6 + self.w27 * f7 + self.w28 * f8 + self.b9)
        f10 = sigmoid(self.w29 * f1 + self.w210 * f2 + self.w211 * f3 + self.w212 * f4 +
                      self.w213 * f5 + self.w214 * f6 + self.w215 * f7 + self.w216 * f8 + self.b10)
        f11 = sigmoid(self.w217 * f1 + self.w218 * f2 + self.w219 * f3 + self.w220 * f4 +
                      self.w221 * f5 + self.w222 * f6 + self.w223 * f7 + self.w224 * f8 + self.b11)
        f12 = sigmoid(self.w225 * f1 + self.w226 * f2 + self.w227 * f3 + self.w228 * f4 +
                      self.w229 * f5 + self.w230 * f6 + self.w231 * f7 + self.w232 * f8 + self.b12)
        f13 = self.w31 * f9 + self.w32 * f10 + self.w33 * f11 + self.w34 * f12 + self.b13  # 第三层传播结果
        return f13

    def loss(slef, y_true, y_pred): # 损失函数，方便进行拟合程度的分析。
        return (y_true - y_pred)**2
    
    def train(self, data, y_trues):  # 训练函数
        learn_rate = 0.01  # 学习率
        epoch = 0  # 迭代次数
        while True:
            loss = 0  # 定义损失量方便后面实时观测到损失函数值。
            for key, value in dictionary.items():  # 前向传播，方便后面梯度下降使用数据。
                x = key                         # 获取训练数据 x为横坐标  y_true 为对应的sin（x）真值
                y_true = value
                f1 = sigmoid(self.w11 * x + self.b1)  # 开始训练的前项传播 # 第一层
                f2 = sigmoid(self.w12 * x + self.b2)
                f3 = sigmoid(self.w13 * x + self.b3)
                f4 = sigmoid(self.w14 * x + self.b4)
                f5 = sigmoid(self.w15 * x + self.b5)
                f6 = sigmoid(self.w16 * x + self.b6)
                f7 = sigmoid(self.w17 * x + self.b7)
                f8 = sigmoid(self.w18 * x + self.b8)
                f9 = sigmoid(self.w21 * f1 + self.w22 * f2 + self.w23 * f3 + self.w24 * f4 +  # 第二层
                             self.w25 * f5 + self.w26 * f6 + self.w27 * f7 + self.w28 * f8 + self.b9)
                f10 = sigmoid(self.w29 * f1 + self.w210 * f2 + self.w211 * f3 + self.w212 * f4 +
                              self.w213 * f5 + self.w214 * f6 + self.w215 * f7 + self.w216 * f8 + self.b10)
                f11 = sigmoid(self.w217 * f1 + self.w218 * f2 + self.w219 * f3 + self.w220 * f4 +
                              self.w221 * f5 + self.w222 * f6 + self.w223 * f7 + self.w224 * f8 + self.b11)
                f12 = sigmoid(self.w225 * f1 + self.w226 * f2 + self.w227 * f3 + self.w228 * f4 +
                              self.w229 * f5 + self.w230 * f6 + self.w231 * f7 + self.w232 * f8 + self.b12)
                f13 = self.w31 * f9 + self.w32 * f10 + self.w33 * f11 + self.w34 * f12 + self.b13   # 第三层
                y_pred = f13  # 计算出样本的预测值
                de_df13 = -2 * (y_true - y_pred)  # 计算梯度   # 误差e对f13的偏导
                df13_df9 = self.w31  # f13对f9-f12的偏导
                df13_df10 = self.w32
                df13_df11 = self.w33
                df13_df12 = self.w34
                df13_dw31 = f9  # f13对w31-w34及 b13的偏导
                df13_dw32 = f10
                df13_dw33 = f11
                df13_dw34 = f12
                df13_db13 = 1
                df9_df1 = self.w21 * f9 * ( 1 - f9 )   # 第二层  # f9对f1-f8的偏导
                df9_df2 = self.w22 * f9 * ( 1 - f9 )
                df9_df3 = self.w23 * f9 * ( 1 - f9 )
                df9_df4 = self.w24 * f9 * ( 1 - f9 )
                df9_df5 = self.w25 * f9 * ( 1 - f9 )
                df9_df6 = self.w26 * f9 * ( 1 - f9 )
                df9_df7 = self.w27 * f9 * ( 1 - f9 )
                df9_df8 = self.w28 * f9 * ( 1 - f9 )
                df9_dw21 = f1 * f9 * ( 1 - f9 )  # f9对w21-w28  b9的偏导
                df9_dw22 = f2 * f9 * ( 1 - f9 )
                df9_dw23 = f3 * f9 * ( 1 - f9 )
                df9_dw24 = f4 * f9 * ( 1 - f9 )
                df9_dw25 = f5 * f9 * ( 1 - f9 )
                df9_dw26 = f6 * f9 * ( 1 - f9 )
                df9_dw27 = f7 * f9 * ( 1 - f9 )
                df9_dw28 = f8 * f9 * ( 1 - f9 )
                df9_db9 = 1
                df10_df1 = self.w29 * f10 * (1 - f10)  # f10对f1-f8的偏导
                df10_df2 = self.w210 * f10 * (1 - f10)
                df10_df3 = self.w211 * f10 * (1 - f10)
                df10_df4 = self.w212 * f10 * (1 - f10)
                df10_df5 = self.w213 * f10 * (1 - f10)
                df10_df6 = self.w214 * f10 * (1 - f10)
                df10_df7 = self.w215 * f10 * (1 - f10)
                df10_df8 = self.w216 * f10 * (1 - f10)
                df10_dw29 = f1 * f10 * (1 - f10)  # f10对w29-w216  b10的偏导
                df10_dw210 = f2 * f10 * (1 - f10)
                df10_dw211 = f3 * f10 * (1 - f10)
                df10_dw212 = f4 * f10 * (1 - f10)
                df10_dw213 = f5 * f10 * (1 - f10)
                df10_dw214 = f6 * f10 * (1 - f10)
                df10_dw215 = f7 * f10 * (1 - f10)
                df10_dw216 = f8 * f10 * (1 - f10)
                df10_db10 = 1
                df11_df1 = self.w217 * f11 * (1 - f11)  # f11对f1-f8的偏导
                df11_df2 = self.w218 * f11 * (1 - f11)
                df11_df3 = self.w219 * f11 * (1 - f11)
                df11_df4 = self.w220 * f11 * (1 - f11)
                df11_df5 = self.w221 * f11 * (1 - f11)
                df11_df6 = self.w222 * f11 * (1 - f11)
                df11_df7 = self.w223 * f11 * (1 - f11)
                df11_df8 = self.w224 * f11 * (1 - f11)
                df11_dw217 = f1 * f11 * (1 - f11)  # df11对w217-w224  b11的偏导
                df11_dw218 = f2 * f11 * (1 - f11)
                df11_dw219 = f3 * f11 * (1 - f11)
                df11_dw220 = f4 * f11 * (1 - f11)
                df11_dw221 = f5 * f11 * (1 - f11)
                df11_dw222 = f6 * f11 * (1 - f11)
                df11_dw223 = f7 * f11 * (1 - f11)
                df11_dw224 = f8 * f11 * (1 - f11)
                df11_db11 = 1
                df12_df1 = self.w225 * f12 * (1 - f12)  # df12对f1-f8的偏导
                df12_df2 = self.w226 * f12 * (1 - f12)
                df12_df3 = self.w227 * f12 * (1 - f12)
                df12_df4 = self.w228 * f12 * (1 - f12)
                df12_df5 = self.w229 * f12 * (1 - f12)
                df12_df6 = self.w230 * f12 * (1 - f12)
                df12_df7 = self.w231 * f12 * (1 - f12)
                df12_df8 = self.w232 * f12 * (1 - f12)
                df12_dw225 = f1 * f12 * (1 - f12)  # df12对w225-w232  b12的偏导
                df12_dw226 = f2 * f12 * (1 - f12)
                df12_dw227 = f3 * f12 * (1 - f12)
                df12_dw228 = f4 * f12 * (1 - f12)
                df12_dw229 = f5 * f12 * (1 - f12)
                df12_dw230 = f6 * f12 * (1 - f12)
                df12_dw231 = f7 * f12 * (1 - f12)
                df12_dw232 = f8 * f12 * (1 - f12)
                df12_db12 = 1
                df1_dw11 = x * f1 * (1 - f1)  # 第一层   # f1-8分别对w11-w18 b1-b8的偏导
                df2_dw12 = x * f2 * (1 - f2)
                df3_dw13 = x * f3 * (1 - f3)
                df4_dw14 = x * f4 * (1 - f4)
                df5_dw15 = x * f5 * (1 - f5)
                df6_dw16 = x * f6 * (1 - f6)
                df7_dw17 = x * f7 * (1 - f7)
                df8_dw18 = x * f8 * (1 - f8)
                df1_db1 = f1 * (1 - f1)
                df2_db2 = f2 * (1 - f2)
                df3_db3 = f3 * (1 - f3)
                df4_db4 = f4 * (1 - f4)
                df5_db5 = f5 * (1 - f5)
                df6_db6 = f6 * (1 - f6)
                df7_db7 = f7 * (1 - f7)
                df8_db8 = f8 * (1 - f8)
                # 更新权值和偏差
                # 第一层
                self.w11 -= learn_rate * (de_df13 * df13_df9 * df9_df1 * df1_dw11 + de_df13 * df13_df10 * df10_df1 * df1_dw11 +
                                          de_df13 * df13_df11 * df11_df1 * df1_dw11 + de_df13 * df13_df12 * df12_df1 * df1_dw11)
                self.w12 -= learn_rate * (de_df13 * df13_df9 * df9_df2 * df2_dw12 + de_df13 * df13_df10 * df10_df2 * df2_dw12 +
                                          de_df13 * df13_df11 * df11_df2 * df2_dw12 + de_df13 * df13_df12 * df12_df2 * df2_dw12)
                self.w13 -= learn_rate * (de_df13 * df13_df9 * df9_df3 * df3_dw13 + de_df13 * df13_df10 * df10_df2 * df3_dw13 +
                                          de_df13 * df13_df11 * df11_df3 * df3_dw13 + de_df13 * df13_df12 * df12_df3 * df3_dw13)
                self.w14 -= learn_rate * (de_df13 * df13_df9 * df9_df4 * df4_dw14 + de_df13 * df13_df10 * df10_df4 * df4_dw14 +
                                          de_df13 * df13_df11 * df11_df4 * df4_dw14 + de_df13 * df13_df12 * df12_df4 * df4_dw14)
                self.w15 -= learn_rate * (de_df13 * df13_df9 * df9_df5 * df5_dw15 + de_df13 * df13_df10 * df10_df5 * df5_dw15 +
                                          de_df13 * df13_df11 * df11_df5 * df5_dw15 + de_df13 * df13_df12 * df12_df5 * df5_dw15)
                self.w16 -= learn_rate * (de_df13 * df13_df9 * df9_df6 * df6_dw16 + de_df13 * df13_df10 * df10_df6 * df6_dw16 +
                                          de_df13 * df13_df11 * df11_df6 * df6_dw16 + de_df13 * df13_df12 * df12_df6 * df6_dw16)
                self.w17 -= learn_rate * (de_df13 * df13_df9 * df9_df7 * df7_dw17 + de_df13 * df13_df10 * df10_df7 * df7_dw17 +
                                          de_df13 * df13_df11 * df11_df7 * df7_dw17 + de_df13 * df13_df12 * df12_df7 * df7_dw17)
                self.w18 -= learn_rate * (de_df13 * df13_df9 * df9_df8 * df8_dw18 + de_df13 * df13_df10 * df10_df8 * df8_dw18 +
                                          de_df13 * df13_df11 * df11_df8 * df8_dw18 + de_df13 * df13_df12 * df12_df8 * df8_dw18)
                self.b1 -= learn_rate * (de_df13 * df13_df9 * df9_df1 * df1_db1 + de_df13 * df13_df10 * df10_df1 * df1_db1 +
                                         de_df13 * df13_df11 * df11_df1 * df1_db1 + de_df13 * df13_df12 * df12_df1 * df1_db1)
                self.b2 -= learn_rate * (de_df13 * df13_df9 * df9_df2 * df2_db2 + de_df13 * df13_df10 * df10_df2 * df2_db2 +
                                         de_df13 * df13_df11 * df11_df2 * df2_db2 + de_df13 * df13_df12 * df12_df2 * df2_db2)
                self.b3 -= learn_rate * (de_df13 * df13_df9 * df9_df3 * df3_db3 + de_df13 * df13_df10 * df10_df3 * df3_db3 +
                                         de_df13 * df13_df11 * df11_df3 * df3_db3 + de_df13 * df13_df12 * df12_df3 * df3_db3)
                self.b4 -= learn_rate * (de_df13 * df13_df9 * df9_df4 * df4_db4 + de_df13 * df13_df10 * df10_df4 * df4_db4 +
                                         de_df13 * df13_df11 * df11_df4 * df4_db4 + de_df13 * df13_df12 * df12_df4 * df4_db4)
                self.b5 -= learn_rate * (de_df13 * df13_df9 * df9_df5 * df5_db5 + de_df13 * df13_df10 * df10_df5 * df5_db5 +
                                         de_df13 * df13_df11 * df11_df5 * df5_db5 + de_df13 * df13_df12 * df12_df5 * df5_db5)
                self.b6 -= learn_rate * (de_df13 * df13_df9 * df9_df6 * df6_db6 + de_df13 * df13_df10 * df10_df6 * df6_db6 +
                                         de_df13 * df13_df11 * df11_df6 * df6_db6 + de_df13 * df13_df12 * df12_df6 * df6_db6)
                self.b7 -= learn_rate * (de_df13 * df13_df9 * df9_df7 * df7_db7 + de_df13 * df13_df10 * df10_df7 * df7_db7 +
                                         de_df13 * df13_df11 * df11_df7 * df7_db7 + de_df13 * df13_df12 * df12_df7 * df7_db7)
                self.b8 -= learn_rate * (de_df13 * df13_df9 * df9_df8 * df8_db8 + de_df13 * df13_df10 * df10_df8 * df8_db8 +
                                         de_df13 * df13_df11 * df11_df8 * df8_db8 + de_df13 * df13_df12 * df12_df8 * df8_db8)
                # 第二层
                self.w21 -= learn_rate * de_df13 * df13_df9 * df9_dw21
                self.w22 -= learn_rate * de_df13 * df13_df9 * df9_dw22
                self.w23 -= learn_rate * de_df13 * df13_df9 * df9_dw23
                self.w24 -= learn_rate * de_df13 * df13_df9 * df9_dw24
                self.w25 -= learn_rate * de_df13 * df13_df9 * df9_dw25
                self.w26 -= learn_rate * de_df13 * df13_df9 * df9_dw26
                self.w27 -= learn_rate * de_df13 * df13_df9 * df9_dw27
                self.w28 -= learn_rate * de_df13 * df13_df9 * df9_dw28
                self.w29 -= learn_rate * de_df13 * df13_df9 * df10_dw29
                self.w210 -= learn_rate * de_df13 * df13_df9 * df10_dw210
                self.w211 -= learn_rate * de_df13 * df13_df9 * df10_dw211
                self.w212 -= learn_rate * de_df13 * df13_df9 * df10_dw212
                self.w213 -= learn_rate * de_df13 * df13_df9 * df10_dw213
                self.w214 -= learn_rate * de_df13 * df13_df9 * df10_dw214
                self.w215 -= learn_rate * de_df13 * df13_df9 * df10_dw215
                self.w216 -= learn_rate * de_df13 * df13_df9 * df10_dw216
                self.w217 -= learn_rate * de_df13 * df13_df11 * df11_dw217
                self.w218 -= learn_rate * de_df13 * df13_df11 * df11_dw218
                self.w219 -= learn_rate * de_df13 * df13_df11 * df11_dw219
                self.w220 -= learn_rate * de_df13 * df13_df11 * df11_dw220
                self.w221 -= learn_rate * de_df13 * df13_df11 * df11_dw221
                self.w222 -= learn_rate * de_df13 * df13_df11 * df11_dw222
                self.w223 -= learn_rate * de_df13 * df13_df11 * df11_dw223
                self.w224 -= learn_rate * de_df13 * df13_df11 * df11_dw224
                self.w225 -= learn_rate * de_df13 * df13_df12 * df12_dw225
                self.w226 -= learn_rate * de_df13 * df13_df12 * df12_dw226
                self.w227 -= learn_rate * de_df13 * df13_df12 * df12_dw227
                self.w228 -= learn_rate * de_df13 * df13_df12 * df12_dw228
                self.w229 -= learn_rate * de_df13 * df13_df12 * df12_dw229
                self.w230 -= learn_rate * de_df13 * df13_df12 * df12_dw230
                self.w231 -= learn_rate * de_df13 * df13_df12 * df12_dw231
                self.w232 -= learn_rate * de_df13 * df13_df12 * df12_dw232
                self.b9 -= learn_rate * de_df13 * df13_df9 * df9_db9
                self.b10 -= learn_rate * de_df13 * df13_df10 * df10_db10
                self.b11 -= learn_rate * de_df13 * df13_df11 * df11_db11
                self.b12 -= learn_rate * de_df13 * df13_df12 * df12_db12
                # 第三层
                self.w31 -= learn_rate * de_df13 * df13_dw31
                self.w32 -= learn_rate * de_df13 * df13_dw32
                self.w33 -= learn_rate * de_df13 * df13_dw33
                self.w34 -= learn_rate * de_df13 * df13_dw34
                self.b13 -= learn_rate * de_df13 * df13_db13
                y_pred = self.feedforward(x)
                loss += self.loss(y_true, y_pred)
            epoch += 1  # 迭代参数计数
            self.epoch.append(epoch)
            self.loss_value.append(loss)
            print("epoch: %d loss:%f " % (epoch, loss/len(y_trues)))  # 打印每次迭代的损失值
            # 如果迭代的损失值小于设定值,假定训练已经完成,此时认为拟合无误，每次迭代为整个训练数据拟合一遍。
            if (loss/len(y_trues)) <= 0.000001:
                return

if __name__ == '__main__':
    N = NeuralNetwork()  # 创建神经网络类实例
    N.train(train_data, y_trues) # 开始训练 将结果保存到网络类中
    print(  # 打印出网络参数
        " w11:%f, w12 :%f, w13 :%f, w14 :%f, w15 :%f, w16 :%f, w17 :%f, w18 :%f,\n\
          w21 :%f, w22 :%f, w23 :%f, w24 :%f, w25 :%f, w26 :%f, w27 :%f, w28 :%f,\n\
          w29 :%f, w210 :%f, w211 :%f, w212 :%f, w213 :%f, w214 :%f, w215 :%f, w216 :%f,\n\
          w217 :%f, w218 :%f, w219 :%f, w220 :%f, w221 :%f, w222 :%f, w223 :%f, w224 :%f,\n\
          w225 :%f, w226 :%f, w227 :%f, w228 :%f, w229 :%f, w230 :%f, w231 :%f, w232 :%f,\n\
          w31 :%f, w32 :%f, w33 :%f, w34 :%f, w35 :%f, w36 :%f, w37 :%f, w38 :%f,\n\
          b1  :%f, b2  :%f, b3  :%f, b4  :%f, b5  :%f, b6  :%f, b7  :%f, b8  :%f,\n\
          b9  :%f, b10 :%f, b11 :%f, b12 :%f, b13 :%f,\n"
        % (N.w11, N.w12, N.w13, N.w14, N.w15, N.w16, N.w17, N.w18,
           N.w21, N.w22, N.w23, N.w24, N.w25, N.w26, N.w27, N.w28,
           N.w29, N.w210, N.w211, N.w212, N.w213, N.w214, N.w215,
           N.w216, N.w217, N.w218, N.w219, N.w220, N.w221, N.w222,
           N.w223, N.w224, N.w225, N.w226, N.w227, N.w228, N.w229,
           N.w230, N.w231, N.w232, N.w31, N.w32, N.w33, N.w34, N.w35,
           N.w36, N.w37, N.w38, N.b1, N.b2, N.b3, N.b4, N.b5,
           N.b6, N.b7, N.b8, N.b9, N.b10, N.b11, N.b12, N.b13
           )
    )
    result1 = []  # 绘制图像
    result2 = []
    result3 = []
    for data in train_data:  # 使用训练集画出真值
        result1.append(np.sin(data))
    for data in train_data:  # 使用训练集在神经网络下的拟合
        result2.append(N.feedforward(data))
    for data in test_data:  # 使用测试集在神经网络下的拟合
        result3.append(N.feedforward(data))
    plot.figure(1)  # 设置图1为结果分析图
    plot.title("Analysis of Result")  # 设置图标题
    plot.plot(train_data, result1, linewidth=1.0, color='r', label="train_data with function of sin")  # 画曲线 设置标签
    plot.plot(train_data, result2, linewidth=1.0, color='b', label="train_data with nerual_net_work")
    plot.plot(test_data, result3, linewidth=1.0, color='g', label="test_data with nerual_net_work")
    plot.legend(loc='best')
    plot.figure(2)  # 设置图2为误差变化图
    plot.title("Change of Loss_value")  # 设置图标题
    plot.plot(N.epoch, N.loss_value, linewidth=1.0, color='k')  # 画曲线 设置标签
    plot.show()  # 显示图像

