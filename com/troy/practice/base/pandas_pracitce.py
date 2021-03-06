import pandas as pd
import numpy as np


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

e = pd.DataFrame(np.arange(24).reshape(4,6))
print(e)
df = pd.read_csv('../../trader/data/SH#600000_01.txt')
print(df)
print(type(df))
print(df.index)
print(df.columns)
print(df.shape)
print(df.dtypes)
print(df.ndim)
print(df.head(3))
print(df.tail())
print(df.tail(3))
print(df.info())
print(df.describe())
df = df.sort_values(by="open",ascending=False)
print(df.tail())
print("*" * 50)
print(df[:20])
print(df['open'])
print(df[:5 ]['close'])
print("*" * 50)
df = pd.DataFrame(np.arange(24).reshape(4,6), index=list("abcd"), columns=list("efghij"))
print(df)
print(df.loc['a','j'])
print(df.iloc[0,5])
print(df.loc[:,'e'])
print(df.loc['a',:])

df = pd.read_csv('../../trader/data/SH#600000_01.txt')
print(df[df['open']>df['close']])

