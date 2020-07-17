#!/anthony/bin/python3
#-*= coding: utf-8 -*-
'''
@FileName : CTC_loss.py
@Time     : 2020/6/26 12:12:50
@Caption  :
           1.y_true:包含真实值标签的张量。类型(samples, max_string_length).
           2.y_pred:包含预测值或softmax输出的张量。类型(samples, time_steps, num_categories)。
           3.input_length:张量(samples, 1)，包含y_pred中每个批处理项的序列长度。
           4.label_length:张量(samples, 1)， 包含y_true中每个批处理项的序列长度。
           最后，返回shape为(samples, 1)的张量，包含每一个元素的CTC损失。
'''
from keras.backend import ctc_label_dense_to_sparse
from pandas.core.ops import array_ops
from sklearn.tests.test_multiclass import perm
from tensorflow_core.python.ops import math_ops
import tensorflow as tf

def CTC_loss(y_true, y_pred, input_length, label_length):
 lable_length = math_ops.to_int32(array_ops.squeeze(label_length))
 input_length = math_ops.to_int32(array_ops.squeeze(input_length))
 sparse_lables = math_ops.to_int32(ctc_label_dense_to_sparse(y_true, lable_length))
 y_pred = math_ops.log(array_ops.transpose(y_pred, perm([1, 2, 3]) + 1e-8))

 return array_ops.expand_dime(
tf.nn.ctc_loss(inputs=y_pred, labels=sparse_lables, sequence_length=input_length), 1)
