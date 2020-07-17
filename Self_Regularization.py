if self.metadata['return_sequences']:
      self.model.add(Flatten())
  self.model.add(Dense(2, kernel_initializer=self.metadata['dense_kernel_initializer'],
                       kernel_regularizer=l1(self.metadata['dense_kernel_regularizer']),
                       activation=self.metadata['dense_activation']))#正则项
  self.model.add(Softmax())


model.add(Dropout(0.5))# 采用50%的dropout，在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元