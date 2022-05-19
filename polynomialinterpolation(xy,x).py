# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 22:10:08 2022

@author: 1
"""

#polynomialinterpolation(xy,x):
    
x=x1  
n=xy.shape[1]#n=11
#print(n)
xy=np.transpose(xy)    #转置
x0=xy[:,[0]]#取第0列,[0]是矩阵形式，0是不知道什么形式（可能是列表
y0=xy[:,[1]]#取第1列

m=len(x)#m=1001
A=np.zeros((n,n)) #n*n全零阵
for q in range(n):#从0开始，n个数
    z=x0[q]
    A[q,0]=1#(出错，不能分配给函数调用),不能用（），要用[]
    for d in range(1,n):#从1开始，n个数
        A[q,d]=A[q,d-1]*z
#print(A)

c=np.multiply(y0,np.transpose(A))
y=[]

for l in range(m):#从0开始，m个数
    z=x[l]
    print(z)
    #M=np.zeros(1,n)  #这种格式错误，需要加()
    M=np.zeros((1,n))
    M[0,0]=1   #注意python从0开始
    
    for d in range(1,n):
        M[0,d]=M[0,d-1]*z   #python不能像matlab那样省略一维数组前面的0
    #y[l]=np.sum(np.multiply(np.transpose(M),c))   
    #这样写会报错，列表超过长度，因为定义的是空列表，不存在索引，要用append
    
    y.append(np.sum(np.multiply(np.transpose(M),c)))
    print(y)