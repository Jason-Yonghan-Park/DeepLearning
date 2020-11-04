##################### Linear Regression 선형 회귀 구현

import tensorflow as tf

# 학습 데이터 구성
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# tf.keras.Sequential 이용해 모델 생성
model = tf.keras.models.Sequential()

# 모델에 레이어 1개 추가
model.add(tf.keras.layers.Dense(1, input_dim=1))

# 학습하기 위해 모델 준비
sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd)

# 모델을 학습
model.fit(x_train, y_train, epochs=100)

# 학습 종료 후 새로운 데이터 적용해 예측
import numpy as np
print(model.predict(np.array([3, 7, 9])))

