############################ Convolutional Neural Network
######################## Fashion MNIST CNN 구현

# 필요한 패키지 임포트
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Fashion MNIST 데이터 적재
fashion_mnist = keras.datasets.fashion_mnist
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
print(train_X.shape, '-', test_X.shape)
print(train_y.shape, '-', test_y.shape)
print(set(train_y))

'''
plt.imshow(train_X[100], cmap='gray')
plt.colorbar()
plt.show()
'''

# 신경망 주입 전 픽셀값 0-1 사이로 정규화
train_X = train_X / 255.0
test_X = test_X / 255.0

# CNN에서 이미지 -> (num, width, height, channel)
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

# class name 지정
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 5*5 이미지 플롯 띄우기
'''
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([]);plt.yticks([])
    plt.grid(False)
    plt.imshow(train_X[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(class_names[train_y[i]])
plt.show()
'''

# sequential 모델 생성 및 레이어 추가
model = tf.keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=16),
    keras.layers.MaxPool2D(strides=(2, 2)),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=32),
    keras.layers.MaxPool2D(strides=(2, 2)),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=64),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(10, activation='softmax')
])

# cost func, loss func 설정
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
history = model.fit(train_X, train_y, epochs=20, validation_split=0.25)

# 모델 검증 및 성능 평가
test_loss, test_acc = model.evaluate(test_X, test_y)
print('evaluation accuracy: {}'.format(test_acc))








