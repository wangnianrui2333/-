# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:38:21 2022

@author: 1
"""

def lagrangeinterpolation(x0,y0,x):

n=len(x0)
m=len(x)

L0=[]
L0=np.zeros((n,1))

for k in range(n):
    L=1
    for j in range(n):
        if j!=k:
            L=L/(x0[k]-x0[j])
    L0[k]=L

y=[]
for l in range(m):
    z=x[l]
    s=0
    for k in range(n):
        L=L0[k]
        for j in range(n):
            if j!=k:
                L=L*(z-x0[j])
        s=s+L*y0[k]
    y.append(s)

return y