################################## Convoultional Neural Network 실습
#################### 손글씨 데이터 셋

# 필요한 패키지 임포트
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 손글씨 데이터 셋 읽어오기
mnist = keras.datasets.mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# 학습 데이터 정규화
train_X = train_X / 255.0
test_X = test_X / 255.0

# 차원 수정
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)

# Sequential 이용해 모델 생성
model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, activation='relu'),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    keras.layers.MaxPool2D(strides=(2, 2)),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=32, activation='relu'),
    keras.layers.MaxPool2D(strides=(2, 2)),
    keras.layers.Conv2D(kernel_size=(3, 3), filters=64, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(rate=0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

# 모델 적합
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.002), metrics=['accuracy'])

# 모델 예측
history = model.fit(train_X, train_y, epochs=20, validation_split=0.25)

# loss, accuracy 시각화
test_loss, test_acc = model.evaluate(test_X, test_y)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.title('loss vs val_loss')
plt.xlabel('epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label = 'val_accuracy')
plt.title('accuracy vs val_accuracy')
plt.xlabel('epochs')
plt.legend()

plt.show()


