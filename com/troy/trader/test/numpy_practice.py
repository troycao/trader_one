import numpy as np
import random

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

