#!/anthony/bin/python3
#-*= coding: utf-8 -*-
'''
@FileName : active_function_maxout.py
@Time     : 2020/6/27 10:09:43
'''

import tensorflow as tf
def max_out(inputs, num_units, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:
       axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of featire({}) is not'
                         'a multipe of num_units({{})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(input, shape), -1, keep_dims= False)
    return outputs
    print(outputs)

