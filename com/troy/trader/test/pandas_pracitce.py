import pandas as pd

a = pd.Series([1,2,3,4,5,6])
print(a)
b = pd.Series([1,2,3,4,5,6],index=list("abcdef"))
print(b)

c = pd.Series({'name':'troy','age':18,'tel':10086})
print(c)
print(c.index)
print(c.values)
print(len(c.index))
print(c['name'])
print(c[1])
for i in c.index:
    print(c[i])
print(list(c.index))
print(list(c.values))
print(list(c.index)[:1])
print(type(c.index))
print(type(c.values))

d = pd.Series(range(6))
print(d)
print(d.where(d>3))
