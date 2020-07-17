'''
Created on 2020年6月15日
author:Anthony_Yu
 compute PRF
'''

from sklearn.metrics import classification_report as cfr
import os
w_true = []#引入数值
fileread = os.listdir('E:/ok/new54.txt')#读文件
for line in fileread.readlines():#依次遍历
    a = w_true.append(line.strip())#删空行
w_pred = []#引入数值
fileread = os.listdir('E:/ok/t59.txt')
for line in fileread.readlines():
    b= w_pred.append(line.strip())

print(cfr(a, b))#使用引用的模块进行计算
