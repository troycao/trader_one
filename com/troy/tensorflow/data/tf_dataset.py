import tensorflow as  tf
import numpy as np
version = tf.__version__
print(version)

# 创建一个dataset
dataset = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7,8,9,0])
print(dataset)

for ele in dataset:
    print(ele.numpy())

dataset = tf.data.Dataset.from_tensor_slices(np.arange(0,24).reshape(4,6))
print(dataset)
for ele in dataset:
    print(ele.numpy())

# 字典创建dataset
dataset = tf.data.Dataset.from_tensor_slices({"name":["troy"],"age":[18],"tel":[10086]})
print(dataset)
for ele in dataset:
    print(ele)

# take获取n个值
dataset = tf.data.Dataset.from_tensor_slices(np.arange(0,24).reshape(4,6))
# dataset = dataset.take(1)
# for ele in dataset:
#     print(ele)

# 重要
# 乱序
dataset = dataset.shuffle(buffer_size=len(dataset))
# 重复
dataset = dataset.repeat(count=3)
# 分批次
dataset = dataset.batch(batch_size=2)
for ele in dataset:
    print(ele.numpy())

dataset = tf.data.Dataset.from_tensor_slices(np.arange(0,9))
print(dataset)
dataset = dataset.map(tf.square)
for ele in dataset:
    print(ele)