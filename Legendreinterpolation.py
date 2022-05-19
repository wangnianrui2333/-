## test3.py

import numpy as np
import time
import matplotlib.pyplot as plt

# 计算n次勒让德多项式函数的m次导数值
def legendre(n,m,x):
    
    if n==0:
        if m>=1:
            return 0
        else:
            return 1
    
    
    s=np.zeros([n+1,m+1])
    
    for j in range(0,m+1):
       if j==0:
          s[0,j]=1
          s[1,j]=x
          for k in range(1,n):
            s[k+1,j]=((2*k+1)*x*s[k,j]-k*s[k-1,j])/(k+1)
          
       else:
           s[0,j]=0
           if j==1:
               s[1,j]=1
           else:
               s[1,j]=0
           
           for k in range(1,n):
               s[k+1,j]=(2*k+1)*s[k,j-1]+s[k-1,j]
           
    r=s[n,m]
    
    return r

# 计算勒让德高斯巴罗托点
def legendregauss(n, m):

    z = list()
    if n == 0 or m >= n:
        z = np.array(z)
        return z

    error = pow(10, -14)
    h = pow(n, -2)
    a = -1
    b = a + h

    z.append(a)

    for k in range(n-m):
        legendre_a = legendre(n, m, a)
        legendre_b = legendre(n, m, b)
        while legendre_a * legendre_b > 0:
            a = b
            legendre_a = legendre_b

            b = a + h
            legendre_b = legendre(n, m, b)

        x = (a+b)/2
        xright = b
        while abs(x-xright) > error:
            xright = x
            x = x - legendre(n, m, x) / legendre(n, m+1, x)
        z.append(x)
        a = x + h
        b = a + h

    z.append(1)

    return np.array(z)

# 计算勒让德插值系数
def legendreinterpolationcoefficients(xi, yi):

    N = len(xi) - 1
    L = np.zeros((N+1, N+1))

    for j in range(N+1):
        x = xi[j]
        k = 0
        L[k, j] = 1
        k = 1
        L[k, j] = x
        for k in range(1, N):
            L[k+1, j] = 1/(k+1) * ((2*k+1)*x*L[k, j] - k*L[k-1, j])

    a = np.zeros((1, N+1))
    for k in range(N):
        for j in range(N+1):
            a[0, k] = a[0, k] + yi[j]*L[k, j]/pow(L[N, j], 2)
        a[0, k] = a[0, k]*(2*k+1)/(N*(N+1))

    for j in range(N+1):
        a[0, N] = a[0, N] + yi[j] / L[N, j]

    a[0, N] = a[0, N]/(N+1)

    return a


# 计算勒让德插值结果
def legendreinterpolation(a, x):
    
    N = len(a[0]) - 1
    M = len(x)

    y = np.zeros((1, M))

    for m in range(M):
        L = np.zeros((1, N+1))
        z = x[m]
        k = 0
        L[0, k] = 1
        k = 1
        L[0, k] = z
        for k in range(1, N):
            L[0, k+1] = 1/(k+1)*((2*k+1)*z*L[0, k]-k*L[0, k-1])
        y[0, m] = np.dot(a, L.transpose())

    return y


time0 = time.time()

N = 60

# 计算勒让德高斯巴罗托点
xi = legendregauss(N, 1)
yi = []
for k in range(len(xi)):
    yi.append(1 / (1 + 25 * pow(xi[k],2)))
yi = np.array(yi)

# 计算勒让德插值系数
a = legendreinterpolationcoefficients(xi, yi)

# 计算勒让德插值结果
x1 = np.arange(-1, 1, 2*pow(10, -4))
y1 = legendreinterpolation(a, x1)

yexact=1/(1+25*x1**2)
error=np.max(np.abs(y1-yexact))


print('error=',error)

time1 = time.time()
cputime = time1 - time0
print('cputime=',cputime)

# 画图
plt.plot(x1, y1[0], ls=':', lw=2, color='r')
plt.show()

