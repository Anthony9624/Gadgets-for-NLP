'''
Created on 2020年5月29日
@author: Anthony_Yu
Add word embeddings to word embeddings
'''

import io
embtxt = "F:/train.txt"
with open(embtxt,"r+") as f:
 content = f.read()
f.seek(0,0)
text = "'PAD'\t"
for i in range(300):
    text += "0\t"
    text = text[0:-1]
f.write(text + '\n' + content)
f.close()
















# a = "你好，我的名字叫"
# b = "Anthony"
# c = a+b
# print(c)

