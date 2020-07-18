#! -*- coding: utf-8 -*-
'''

author:BoJone

'''
import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf

# 判断是tf.keras还是纯keras的标记
is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K


class AdaX_V1(keras.optimizers.Optimizer):
    """AdaX_V1优化器（纯Keras版）
    """
    def __init__(
        self, learning_rate=0.001, beta_1=0.9, beta_2=0.0001, **kwargs
    ):
        self.initial_decay = kwargs.pop('decay', 0.0)
        self.epsilon = kwargs.pop('epsilon', K.epsilon())
        learning_rate = kwargs.pop('lr', learning_rate)
        super(AdaX_V1, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.learning_rate = K.variable(learning_rate, name='learning_rate')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(self.initial_decay, name='decay')

    @K.symbolic
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate
        if self.initial_decay > 0.0:
            lr = lr * (
                1.0 / (
                    1.0 +
                    self.decay * K.cast(self.iterations, K.dtype(self.decay))
                )
            )

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * K.sqrt(K.pow(1.0 + self.beta_2, t) - 1.0)

        ms = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p), name='m_' + str(i))
            for (i, p) in enumerate(params)
        ]
        vs = [
            K.zeros(K.int_shape(p), dtype=K.dtype(p), name='v_' + str(i))
            for (i, p) in enumerate(params)
        ]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = self.beta_1 * m + (1.0 - self.beta_1) * g
            v_t = (1.0 + self.beta_2) * v + self.beta_2 * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {
            'learning_rate': float(K.get_value(self.learning_rate)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'epsilon': self.epsilon,
        }
        base_config = super(AdaX_V1, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))