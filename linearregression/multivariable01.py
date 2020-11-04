######################## Multi Variable Linear Regression

import tensorflow as tf
import numpy as np

# 학습 데이터
x_data = np.array([[73., 80., 75.], [93., 88., 93.], [89., 91., 90.], [96., 98., 100.], [73., 66., 70]])
y_data = np.array([[152.], [185.], [180.], [196.], [142.]])

# Sequential 통해 모델 생성
model = tf.keras.models.Sequential()

# 모델에 레이어 추가 - 다중레이어 추가
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(3, )))
# 모델에 레이어 2개 추가
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(1))

# 모델 준비
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))

# 모델 학습
model.fit(x_data, y_data, epochs=100)

# 새로운 데이터 모델 적용 후 예측
model.predict(np.array([[79., 83., 91.]]))






