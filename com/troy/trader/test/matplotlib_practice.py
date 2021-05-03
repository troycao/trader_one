from matplotlib import pyplot as plt


x = range(2,26,2)
y = [15,14.5,17,18,19,20,25,24,23,18,17,14]
print(x)
print(y)
plt.figure(figsize=(40,16 ),dpi=80)
# 折线图 主要是看数据趋势和关联
plt.plot(x,y)
plt.show()
# 设置标签
plt.xlabel()
plt.ylabel()
# 设置刻度
plt.xticks()
plt.yticks()
#保持tupian
plt.imsave()
# 柱状图 离散的数据
plt.bar()
plt.barh()

# 柱状图 连续的数据
plt.hist()
# 散点图
plt.scatter(x,y)

