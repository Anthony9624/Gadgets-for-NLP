import numpy as np
##设置全部数据，不输出省略号
import sys
np.set_printoptions(threshold=sys.maxsize)
boxes = np.load('D:/testlabel.npy')
print(boxes)
np.savetxt('D:/111(1).txt', boxes, fmt='%s', newline='\n')
print('---------------------boxes--------------------------')
# a = np.loadtxt('E:/10(1).txt')
# np.save('E:/vec.npy',a)
# print(a)