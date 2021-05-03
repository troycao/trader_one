import random

import numpy as np

# np生成数组
a = np.array([1,2,3])
print(a)
print(type(a))
b = np.array(range(1,6))
print(b)
c = np.arange(1,9,2)
print(c)
print(c.dtype)
d = np.arange(1,100,dtype=float)
print(d)
print(d.dtype)

e = np.array([1,0,1,0,1], dtype=bool)
print(e)
f = e.astype(int)
print(f)

g = np.array([random.random() for i in range(100)])
print(g)

#######npmpy 形状
a = np.arange(10)
print(a)
print(a.shape)

b = np.array([[1,2,3],[4,5,6]])
print(b)
print(b.shape)

c = np.array([[[1,2,3],
               [4,5,6]],
              [[7,8,9],
               [10,11,12]]])
print(c)
print(c.shape)

d = np.arange(12)
print(d.shape)
print(d)
e = d.reshape(3,4)
print(e)
print(e.shape)
e = d.reshape(2,2,3)
print(e)
print(e.shape)

g = e.reshape((4,3))
print(g)
print(g.reshape(12))
print(e)

print(g.reshape((12,1)))
print(g.flatten())

print(g + 2)

h = np.arange(0,24).reshape(4,6)
print(h)
i = np.arange(100,124).reshape(4,6)
print(i)

print(h + i)
print(h * i)
print(i -h)
print(h/i)

### 转置
a = np.arange(0, 24).reshape(4,6)
print(a)
b = a.transpose()
print(b)
print(a.T)
print(a.swapaxes(1,0))

# np
t1 = np.loadtxt('../data/SH#600000_01.txt', delimiter=" ", dtype=np.int_)
print(t1)
print(t1[0])
print(t1[[1,3]])

###读取
a = np.arange(0,24).reshape(4,6)
# print(a)
# print(a[2,:])
# print(a[1:4,1:4])
# print(a[[0,2],[0,2]])
# print(a[[0,1],:])

# print(a<10)
# print(a[a>20])

a[a<10] = 5
print(a)

a = 3 if 4>3 else 4
print(a)

a = np.arange(0,24).reshape(4,6)
a = np.where(a>10,10,-1)
print(a)

# 拼接
a = np.arange(0, 24).reshape(4,6)
print(a)
b = np.arange(100,124).reshape(4,6)
print(b)

c = np.vstack((a,b))
print(c)
print(c.shape)
d = np.hstack((a,b))
print(d)
print(d.shape)

# a = 1
# b=2
# a,b=b,a
#
# print(a)
# print(b)
print(a)
a[[1,2],:] = a[[2,1],:]
print(a)
print(b)
b[:,[1,2]] = b[:,[2,1]]
print(b)

print(np.zeros((2,3)))
print(np.ones((3,4)))
print(np.eye(10))

a = np.eye(10)
print(np.nanargmax(a, axis=1))
print(np.nanargmin(a, axis=1))
a[a==1] = -1
print(a)
print(np.nanargmin(a, axis=1))

a = np.random.rand(2,3)
b = np.random.randn(2,3)
c = np.random.randint(10,20,(2,3))
d = np.random.randint(2,5,10)
print(a)
print(b)
print(c)
print(d)

d = np.random.randint(1,20,(2,5))
print(d)

np.random.seed(10)
a = np.random.randint(1,20,(2,5))
print(a)

a =  np.arange(24).reshape(4,6).astype(np.float32)
print(a)
a[2,3:] = np.nan
print(a)
