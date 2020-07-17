#!/anthony/bin/python3
#-*= coding: utf-8 -*-
'''
@FileName : NormailzeMult.py
@Time     : 2020/7/6 14:53:13
@Author   : Anthony_Yu
@Contact  : yubochina@aliyun.com
'''
import numpy as np
def NormailzeMult(data):
    data = np.array(data)
    normailze = np.arange(data.shape[1],data_type='float64')
    normailze = normailze.reshape(data.shape[1],2)
    print(normailze.shape)
    for i in range(0,data.shape[1]):
        list = data[:,i]
        listlow,listhigh = np.percentile(list,[0,100])
        normailze[i,0] = listlow
        normailze[i:1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            for j in range(0,data.shape[0]):
                data[j,i] = (data[j,i] - listlow)/delta
    return data,normailze
