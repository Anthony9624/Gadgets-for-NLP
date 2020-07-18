#!/anthony/bin/python3
#-*= coding: utf-8 -*-
'''
@FileName : FNormailzeMult.py
@Time     : 2020/7/18 9:16:50
@Author   : Anthony_Yu
@Contact  : yubochina@aliyun.com
'''
import numpy as nps
def FNormalizeMult(datas,Normalize):
    data = nps.array(datas)
    for i in  range(0,datas.shape[1]):
        listlower =  Normalize[i,0]
        listhigher = Normalize[i,1]
        deltas = listhigher - listlower
        if deltas != 0:
            #第j行
            for j in range(0,datas.shape[0]):
                data[j,i]  =  datas[j,i]*deltas + listlower

    return data