import os
#import numpy
#import sys

def P2toP3(tname, p2top3FileName):
    if os.path.exists(tname):
        for tranpath, trannames, filenames in os.walk(tname):
            for filename in filenames:
                if filename.endswith('.py'):
                    fileFullName = os.path.join(tranpath, filename)
                    print('Processing File:', fileFullName)
                    pycode2topy3 = ("python " + p2top3FileName + " -w " +
                                  fileFullName)
                    print((os.popen(pycode2topy3, 'r').read()))


# tranname即为需要转换的文件目录
tranname = "C://Users//Ant//OneDrive//桌面//触发词//attentionkk"

# p2top3FileName即为Python安装目录下的2to3.py的路径。
p2top3FileName = ("F://An//envs//keras//scripts//2to3.py")

P2toP3(tranname, p2top3FileName)