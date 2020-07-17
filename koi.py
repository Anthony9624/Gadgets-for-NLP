'''
Created on 2020年6月13日
Keras_lstm(Blstm)
'''

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import Embedding
from keras.layers import LSTM
from keras.regularizers import l1#加入正则表达式
import numpy as np
import os
from numpy import argmax as ag
import pandas as pd


data_path=""#为文件路径的文件夹
def load_data():
 train_path = os.path.join(data_path, "ptb.train.txt")
 test_path = os.path.join(data_path, "ptb.test.txt")

# #读文件(借鉴)
# train_data = pd.read_csv("", header=None)
# train_label = pd.read_csv("", header=None)
# test_data = pd.read_csv("", header=None)
# test_label = pd.read_csv("", header=None)
# x_train = np.array(train_data.iloc[:,0:29])
# y_train = np.array(train_label.iloc[:,0:8])
# x_test = np.array(test_data.iloc[:,0:29])
# y_test = np.array(test_label.iloc[:,0:8])

#若两个数据不一样，则对数据进行填充 填充为相同的形状
# max_len=500
# train_data = tf.keras.preprocessing.sequence.pad_sequences(x_train,value =1000,maxlen=max_len,padding='post')
# test_data = tf.keras.preprocessing.sequence.pad_sequences(x_test,value=0,maxlen=max_len,padding='post')
model = Sequential()#序贯式模型

model.add(Dense(500,input_shape=(784,))) # 输入层，28*28=784
'''
# Dense为全连接层，；
# 例：500表示输出的维度，完整的输出表示：(*,500)：即输出任意个500维的数据流；
# input_shape(784,) 表示输入维度是784，完整的输入表示：(*,784)：即输入N个784维度的数据 
'''
def keras_lstm(maxlen=2048, maxfea=1116, self=20,batch_size = 128 ):#构建网络层
 model.add((Embedding(max_features=maxfea, input_length=maxlen, batch_size=batch_size)))
 model.add(Dense(500, Activation='tanh')) # 隐藏层节点500个
 model.add(Activation('reul'))# 激活层Activation，激活函数是reul
 # model.add(Dropout(0.5))# 采用50%的dropout，在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，防止过拟合。
 model.add(Dense(1)) # 输出结果是10个类别，十个输出节点，所以维度是10
 model.add(LSTM(256,return_sequences=True, dropout=0.5, recurrent_dropout=0.2))# 采用50%的dropout
 model.add(Activation('softmax')) # 最后一层用softmax作为激活函数
 sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
 '''
 # 优化函数，设定学习率（lr）参数
 #decay为学习率衰减因子
 #利用牛顿动量法(nesterov)进行优化, 
 #设置基于小批量梯度下降来实现的Momentum（动量）
 '''
 model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode='categorical')  # 编译模型并使用交叉熵作为loss函数
 print("Hold on,training...")
'''
激活函数：tanh
损失函数：分类_交叉熵
优化器：sgd
'''
#训练
# if __name__ =="__main__":
#  score, acc = model.evaluate(x_test, y_test)
#  print('Test score:', score)
#  print('Test accuracy:', acc)
#  for i,data in enumerate(x_test):
#    res = model.predict(data)
#    print (ag(res))




























'''
备用：自身正则
# if self.metadata['return_sequences']:
 #     self.model.add(Flatten())
 # self.model.add(Dense(2, kernel_initializer=self.metadata['dense_kernel_initializer'],
 #                      kernel_regularizer=l1(self.metadata['dense_kernel_regularizer']),
 #                      activation=self.metadata['dense_activation']))#正则项
 # self.model.add(Softmax())


model.add(Dropout(0.5))# 采用50%的dropout，在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元



备用：
#双层双向LSTM
def keras_Blstm(num_words=10000,max_len=500):
    embedding_dim = 16
    batch_size = 128
    model = tf.keras.models.Sequential([
        # input_dim为词汇表的大小  output_dim为输出embedding压缩后的维度   input_length为输入的长度
        keras.layers.Embedding(num_words, embedding_dim, input_length=max_len),
        # batch_size * max_length * embedding_dim
        #   -> batch_size * embedding_dim
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=64, return_sequences=True), ),
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=64, return_sequences=False), ),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, y_train,
                        epochs=30,
                        batch_size=batch_size,
                        validation_split=0.2)


























'''