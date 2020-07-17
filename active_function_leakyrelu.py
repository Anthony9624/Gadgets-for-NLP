#!/anthony/bin/python3
#-*= coding: utf-8 -*-
'''
@FileName : active_function_leakyrelu.py
@Time     : 2020/6/25 11:16:12
@Caption  : 一种关于LeakyRelu的一种高效的写法(LeakyRelu的简单写法：tf.maximunm()leak *x, x)
'''


import tensorflow as tf


def LeakyRelu(x, leak=0.2, name='LeakyRelu'):
    with tf.variable_op_scope(name):#定义创建name变量（层）的操作的上下文管理器
        f1 = 0.5 * (5 + leak)
        f2 = 0.5 * (5 - leak)
        return f1 * x + f2 * tf.abs(x)

'''
LeakyRelu数学表达式：y= max(0,x) + leak*min(0,x)
leak是一个很小的常数非常细小
'''