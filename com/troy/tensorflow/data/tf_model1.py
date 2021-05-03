import tensorflow as tf

# 第一步：准备训练数据
# 加载数据
(train_images,train_labels),(test_images,test_labels) =tf.keras.datasets.mnist.load_data()
print(train_images.shape)
train_images = train_images/255
test_images = test_images/255

print(len(train_labels))
print(len(test_images))
print(train_images.shape)

# 将数据转换为tf.data.DateSet
dataset_train_images = tf.data.Dataset.from_tensor_slices(train_images)
print(dataset_train_images)
dataset_train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
print(dataset_train_labels)

# 合并image和label
dataset_train = tf.data.Dataset.zip((dataset_train_images,dataset_train_labels))
dataset_test = tf.data.Dataset.from_tensor_slices((test_images,test_labels))
dataset_test = dataset_test.batch(64)
print(dataset_train)
# 乱序、重复、分批次
dataset_train = dataset_train.shuffle(10000).repeat().batch(64)

# 第二步：建立model
# 创建一个线性的模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 第三部：编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

# 第四部：训练
steps_per_epoch = train_images.shape[0]//64
validation_steps = test_images.shape[0]//64
histroy = model.fit(dataset_train,
                    epochs=5,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=dataset_test,
                    validation_steps=validation_steps)
print(histroy)