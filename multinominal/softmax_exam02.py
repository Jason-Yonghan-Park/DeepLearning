########################### MNIST Dataset 실습
############ 손글씨 MNIST 숫자 분류

# 필요한 패키지 임포트
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import matplotlib.pyplot as plt

# 손글씨 MNIST 패키지 적재
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 자료 구조 확인
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)
print(set(y_train)) #{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

# 픽셀값 범위 0-1 사이로 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 손글씨 MNIST 이미지 5*5 플롯
'''
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()
'''

# Sequential 모델 생성 및 레이어 추가
model = tf.keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# cost func, loss func 설정
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.005), metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train, y_train, epochs=20, validation_split=0.25)

# 테스트 데이터셋 -> 모델 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print('evaluate accuracy: {}'.format(test_acc))

# loss, val_loss, accuracy, val_accuracy 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.title('loss vs val_loss')
plt.xlabel('Epochs')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.title('accuracy vs val_accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

