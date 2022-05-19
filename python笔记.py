# -*- coding: utf-8 -*-
"""
笔记
"""


"""
np运算
np.multiply(a,b)
矩阵对应位置元素相乘

no.dot(a,b)
点乘

np.matrmul(a,b)
矩阵相乘

np.sum(a,b)
矩阵对应位置元素相加

np.power(a,b)
a^b

np.linalg.norm 求范数

"""



"""
1.列表转数组
import numpy as np
x = [1,2,3,4]
y = np.array(x)

2.列表转矩阵
import numpy as np
x = [1,2,3,4]
y = np.mat(x)

3.数组转列表
y.tolist()  # y : numpy.array

4.数组转矩阵
 np.mat(y) ## y : numpy.array
 
 5.矩阵转列表
 z.tolist() # z: numpy.mat
 
 6.矩阵转数组
 np.array(z)  # z :numpy.mat
"""


"""
获取矩阵
行数：
a.shape[0]
列数：
a.shape[1]
"""


"""
矩阵转置：
a.T
c=transpose(a)
"""



"""
将二元数组转化为一元数组：
y=y.flatten()
"""



"""
 array = numpy.linspace(start, end, num=num_points)
 将在start和end之间生成一个统一的序列，共有num_points个元素。

包含end
"""

"""
range(start, stop[, step])

参数说明：
start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）;
stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
step：步长，默认为1。例如：range（0， 5） 等价于 range(0, 5, 1)

实例>>>range(10) # 从 0 开始到 10
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>>> range(1, 11) # 从 1 开始到 11
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
>>> range(0, 30, 5) # 步长为 5
[0, 5, 10, 15, 20, 25]
>>> range(0, 10, 3) # 步长为 3
[0, 3, 6, 9]
>>> range(0, -10, -1) # 负数
[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
>>> range(0)
"""


"""
若步长为小数：
arange([start,] stop[, step,], dtype=None)
根据start与stop指定的范围以及step设定的步长，生成一个 ndarray。 
dtype : dtype

  np.arange(3)  
  array([0, 1, 2])  
  >>> np.arange(3.0)  
  array([ 0.,  1.,  2.])  
  >>> np.arange(3,7)  
  array([3, 4, 5, 6])  
  >>> np.arange(3,7,2)  
  array([3, 5])  
  
arange(0,1,0.1)  
array([ 0. ,  0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9]) 


"""


"""
zeros函数：
用法：zeros(shape, dtype=float, order='C')
返回：返回来一个给定形状和类型的用0填充的数组；
参数：shape:形状
            dtype:数据类型，可选参数，默认numpy.float64
            dtype类型：t ,位域,如t4代表4位
                                 b,布尔值，true or false
                                 i,整数,如i8(64位）
                                u,无符号整数，u8(64位）
                                f,浮点数，f8（64位）
                               c,浮点负数，
                                o,对象，
                               s,a，字符串，s24
                               u,unicode,u24
            order:可选参数，c代表与c语言类似，行优先；F代表列优先
例子：
np.zeros(5)
array([ 0.,  0.,  0.,  0.,  0.])

np.zeros((5,), dtype=np.int)
array([0, 0, 0, 0, 0])

np.zeros((2, 1))
array([[ 0.],
       [ 0.]])

s = (2,2)
np.zeros(s)
array([[ 0.,  0.],
       [ 0.,  0.]])

np.zeros((2,), dtype=[('x', 'i4'), ('y', 'i4')]) # custom dtype
array([(0, 0), (0, 0)],
      dtype=[('x', '<i4'), ('y', '<i4')])
"""



"""
矩阵取元素：
a[:,1]:表示取a中所有行的第1列的数值<class 'numpy.int32'>
a[:,[1]]:取出的是矩阵形式，便于矩阵运算<class 'numpy.ndarray'>
a[:,:-1]：最后一列不取
"""



"""
添加元素：
数组、矩阵：
b=row_stack((b,e))
c=column_stack((c,e))

列表：
x.append()


"""



"""
concatenate拼接矩阵
 a=np.array([1,2,3])
>>> b=np.array([11,22,33])
>>> c=np.array([44,55,66])
>>> np.concatenate((a,b,c),axis=0)  # 默认情况下，axis=0可以不写
array([ 1,  2,  3, 11, 22, 33, 44, 55, 66]) #对于一维数组拼接，axis的值不影响最后的结果 
 
>>> a=np.array([[1,2,3],[4,5,6]])
>>> b=np.array([[11,21,31],[7,8,9]])
>>> np.concatenate((a,b),axis=0)
array([[ 1,  2,  3],
       [ 4,  5,  6],
       [11, 21, 31],
       [ 7,  8,  9]])
 
>>> np.concatenate((a,b),axis=1)  #axis=1表示对应行的数组进行拼接
array([[ 1,  2,  3, 11, 21, 31],
       [ 4,  5,  6,  7,  8,  9]])

"""



"""
import matplotlib.pyplot as plt
调用 subplot() 函数可以创建一个子图，然后程序就可以在子图上进行绘制。
subplot(nrows, ncols, index, **kwargs) 函数的 
nrows 参数指定将数据图区域分成多少行；
ncols 参数指定将数据图区域分成多少列；
index 参数指定获取第几个区域。
subplot() 函数也支持直接传入一个三位数的参数，
其中第一位数将作为 nrows 参数；
第二位数将作为 ncols 参数；
第三位数将作为 index 参数。
"""



"""
plt.plot(x, y, ls='-', lw=2, label='xxx', color='g' )
x： x轴上的值
y： y轴上的值
ls：线条风格 (linestyle)
lw：线条宽度 (linewidth)
label：标签文本

import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0.5, 10, 1000)
y = np.cos(x)
plt.plot(x, y, ls='-', lw=2, label='cosine', color='purple')
plt.legend()
plt.xlabel('independent variable')
plt.ylabel('dependent variable')
plt.show()

"""



"""
向numpy列表（数组）中加入元素：
xi=np.append(xi,1)
xi=np.append(-1,xi)
"""



"""
排序
L.sort(*, key=None, reverse=None)
*	迭代类型的数据列表
key	函数类型，比较的原则
reverse	为 True 时逆序
"""



"""
random模块：

random.random()
返回一个随机的浮点数，其在0至1的范围之内

random.seed(a=None, version=2)：
指定种子来初始化伪随机数生成器。

random.randrange(start, stop[, stop])：
返回从 start 开始到 stop 结束、步长为 step 的随机数。
其实就相当于 choice(range(start, stop, step)) 的效果，
只不过实际底层并不生成区间对象。

random.randint(a, b)：
生成一个范围为 a≤N≤b 的随机数。其等同于 randrange(a, b+1) 的效果。

random.choice(seq)：
从 seq 中随机抽取一个元素，如果 seq 为空，则引发 IndexError 异常。

random.choices(seq, weights=None, cum_weights=None, k=1)：
从 seq 序列中抽取 k 个元素，还可通过 weights 指定各元素被抽取的权重（代表被抽取的可能性高低）。

random.shuffle(x[, random])：对 x 序列执行洗牌“随机排列”操作。

random.sample(population, k)：从 population 序列中随机抽取 k 个独立的元素。

random.random()：生成一个从0.0（包含）到 1.0（不包含）之间的伪随机浮点数。

random.uniform(a, b)：生成一个范围为 a≤N≤b 的随机数。

random.expovariate(lambd)：生成呈指数分布的随机数。其中 lambd 参数(其实应该是 lambda，只是 lambda 是 Python 关键字，所以简写成 lambd）为 1 除以期望平均值。如果 lambd 是正值，则返回的随机数是从 0 到正无穷大；如果 lambd 为负值，则返回的随机数是从负无穷大到 0。

"""



"""
取整：
向上取整：math.ceil(-0.5)=0
四舍五入：round(-2.5)=-2
    当末尾的5的前一位为奇数：向绝对值更大的方向取整（比如-1.5、1.5处理结果）；
    当末尾的5的前一位为偶数：去尾取整（比如-2.5，-0.5，0.5和2.5的处理结果）
向下取整：math.floor(0.9)=0
向零取整：int(0.9)=0
整除：(-1)//2=-1  ,将结果向下取整

"""