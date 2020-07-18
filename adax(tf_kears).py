#! -*- coding: utf-8 -*-
'''

author:BoJone

'''
import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf
class AdaX_V2(keras.optimizers.Optimizer):
    """重新定义AdaX_V2优化器，便于派生出新的优化器
    （tensorflow的optimizer_v2类）
    """
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.0001,
        epsilon=1e-6,
        **kwargs
    ):
        kwargs['name'] = kwargs.get('name') or 'AdaX_V2'
        super(AdaX_V2, self).__init__(**kwargs)
        self._set_hyper('learning_rate', learning_rate)
        self._set_hyper('beta_1', beta_1)
        self._set_hyper('beta_2', beta_2)
        self.epsilon = epsilon or K.epislon()

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    def _resource_apply(self, grad, var, indices=None):
        # 准备变量
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, 'm')
        v = self.get_slot(var, 'v')
        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)
        epsilon_t = K.cast(self.epsilon, var_dtype)
        local_step = K.cast(self.iterations + 1, var_dtype)

        # 更新公式
        if indices is None:
            m_t = K.update(m, beta_1_t * m + (1 - beta_1_t) * grad)
            v_t = K.update(v, (1 + beta_2_t) * v + beta_2_t * grad**2)
        else:
            mv_ops = [
                K.update(m, beta_1_t * m),
                K.update(v, (1 + beta_2_t) * v)
            ]
            with tf.control_dependencies(mv_ops):
                m_t = self._resource_scatter_add(
                    m, indices, (1 - beta_1_t) * grad
                )
                v_t = self._resource_scatter_add(v, indices, beta_2_t * grad**2)

        # 返回算子
        with tf.control_dependencies([m_t, v_t]):
            v_t = v_t / (K.pow(1.0 + beta_2_t, local_step) - 1.0)
            var_t = var - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)
            return K.update(var, var_t)

    def _resource_apply_dense(self, grad, var):
        return self._resource_apply(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        return self._resource_apply(grad, var, indices)

    def get_config(self):
        config = {
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'epsilon': self.epsilon,
        }
        base_config = super(AdaX_V2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
