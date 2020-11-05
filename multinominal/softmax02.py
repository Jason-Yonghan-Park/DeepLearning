############################### Fashion MNIST Dataset

# Fashion MNIST 분류
# 필요한 패키지 임포트
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Fashion MNIST -> keras.datasets
fashion_mnist = keras.datasets.fashion_mnist
(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

# 학습 데이터 60000개, 테스트 데이터 10000개, 클래스 10개
print(train_X.shape, test_X.shape)
print(set(train_y))

# 첫번째 이미지 화면 출력
plt.imshow(train_X[0], cmap='gray')
#plt.show()

# 이미지 클래스 이름 출력 위해 별도 변수에 저장
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 신경말 모델 주입 전 픽셀값 범위 0-1사이로 조정해 정규화
# 학습데이터셋 / 테스트데이터셋 255로 나누어 정규화
train_X = train_X / 255.0
test_X = test_X / 255.0

# Fashion MNIST 이미지 5*5 플롯 띄우기
"""
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # x축 y축 눈금 삭제
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 컬러맵 흑백으로 설정
    plt.imshow(train_X[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_y[i]])
plt.show()
"""

# Sequential 모델 생성 및 레이어 설정
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# cost func, loss func 설정
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')

# 모델 학습
history = model.fit(train_X, train_y, epochs=20, validation_split=0.25)

# 테스트 데이터셋 -> 모델 성능 평가
test_loss, test_acc = model.evaluate(test_X, test_y)
print('evaluate accuracy: {}'.format(test_acc))

# history 시각화
"""
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.title('loss vs val_loss')
plt.xlabel('epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r--', label='val_accuracy')
plt.title('accuracy vs val_accuracy')
plt.xlabel('epochs')
plt.legend()

plt.show()
"""
# 테스트 데이터 사용해 예측하고 그 결과 시각화
pre = model.predict(test_X)
print(pre[0])
print([round(p, 3) for p in pre[0]])
print(np.argmax(pre[0]), '-', class_names[np.argmax(pre[0])])
plt.imshow(test_X[0], cmap=plt.cm.binary)
plt.show()

# 모델 학습 및 예측 결과에 대한 10개 클래스 신뢰도 시각화

# 이미지 정보 받아 현재 인덱스 해당하는 이미지 차트로 출력하는 함수
def plot_img(i, pre_arr, true_label, img):
    # 현재 인덱스에 해당하는 예측 값, 정답, 이미지 정보 읽음
    pre_arr, true_label, img = pre_arr[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([]);plt.yticks([])
    # 이미지 출력
    plt.imshow(img, cmap=plt.cm.binary)
    # 배열에서 제일 큰 값 -> 예측한 값 가지고 옴
    pre_label = np.argmax(pre_arr)
    if pre_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[pre_label], 100*np.max(pre_arr), class_names[true_label]), color=color)

# 예측 결과 카테고리별로 세로 막대 그래프 출력하는 함수
def plot_value(i, pre_arr, true_label):
    pre_arr, true_label = pre_arr[i], true_label[i]
    plt.grid(False)
    plt.xticks([]);plt.yticks([])

    barplot = plt.bar(range(10), pre_arr, color='#777777')
    plt.ylim([0, 1])
    pre_label = np.argmax(pre_arr)
    barplot[pre_label].set_color('red')
    barplot[pre_label].set_color('blue')

# 가로 세로 출력할 결과의 개수
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_img(i, pre, test_y, test_X)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value(i, pre, test_y)

#plt.show()

img = np.expand_dims(test_X[100], 0)
pred = model.predict(img)
plot_value(0, pred, [test_y[100]])
plt.xticks(range(10), class_names, rotation=45)
plt.show()









