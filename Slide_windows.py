# -*- encoding: utf-8 -*-
"""
@File    : Slide_Windows.py
@Time    : 2020/9/20 23:11
@Author  : Anthony_yu
@Email   : yubochina@aliyun.com
@Software: PyCharm
"""
class CollectionUtil:
    def window(idx,arraySize,windowSize,containsCenterIdx=True): # 实现滑动窗口
        """
        获得当前位置的滑动窗口[元素的下标数组]
        -----------------------------------
        + idx : 当前位置下标
        + containsCenterIdx : 返回结果中，是否需要包含idex索引本身
        + 获得长为arraySize的列表中，以idex为中心，前后分别长windowSize个元素的的滑动窗口的元素下标数组
        + 默认数组下标最小为0
        + 滑动窗口总长 2*windowSize+1
        """
        if idx>=arraySize or idx < 0 or arraySize<1:
            raise ValueError("idx '",idx,"' out of arraySize '",arraySize,"' or them is error value!");
        if 2*windowSize+1 > arraySize:
            raise ValueError("2*windowSize+1 > arraySize! [ windowSize:",windowSize," | arraySize:",arraySize," ]");
        window = [];
        leftStart = (idx-windowSize)%(arraySize-1);  # 1,10,3 -> 7,8,9,0,1,2,3
        rightEnd = (idx+windowSize)%(arraySize-1);  # 9  0  1  2
        isRightWindowContinuous = True if idx+windowSize==rightEnd else False; # 判断右半窗口是否连贯
        for i in range(leftStart,leftStart + windowSize): # range(m,n) = [m,n)
            window.append(i);
            pass;
        if containsCenterIdx == True:
            window.append(idx);
        if isRightWindowContinuous == True:
            for i in range(rightEnd-windowSize+1,rightEnd+1): # 2-4+1
                window.append(i);
                pass;
        else : # 不连贯 即 右半窗口，一部分在数组开头，一部分在数组结尾
            for i in range(idx+1,arraySize-1+1):
                window.append(i);
                pass;
            for i in range(0,rightEnd):
                window.append(i);
                pass;
        return window;

print(CollectionUtil.window(2,10,3)); # [7, 8, 9, 1, 2, 3, 4]
# print(CollectionUtil.window(9,10,3)); # [6, 7, 8, 9, 0, 1, 2]
# print(CollectionUtil.window(8,10,4)); # [4, 5, 6, 7, 8, 9, 0, 1, 2]
# print(CollectionUtil.window(7,10,4)); # [3, 4, 5, 6, 7, 8, 9, 0, 1]
# print(CollectionUtil.window(3,7,2)); # [1,2,3,4,5]
# print(CollectionUtil.window(0,7,2)); # [5,6,0,1,2]
print(CollectionUtil.window(6,7,2)); # [4,5,6,0,1]
