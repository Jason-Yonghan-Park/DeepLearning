################## Perceptron 퍼셉트론 구현
################# AND Gate
import tensorflow as tf
import numpy as np

# AND 연산에 필요한 데이터 준비
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]])

# tf.keras.models.Sequential 이용해 모델 생성
# 뉴런+뉴런 = 레이어 -> 여러 겹 배치 -> 다중 레이어 신경망 구조
model = tf.keras.models.Sequential()

# 신경망 레이어 추가 - 단층 perceptron 구성
model.add(tf.keras.layers.Dense(1, activation='linear', input_shape=(2, )))

# 모델 준비 및 학습
model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.mse, metrics=['acc'])
model.fit(x, y, epochs=50)

# 학습 이후 모델이 가지고 있는 값 출력
print(model.get_weights())

# 새로운 데이터 모델 이용해 예측
print(model.predict(np.array([[1, 1], [0, 1], [1, 0], [0, 0]])))

