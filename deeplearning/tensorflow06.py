################## Multi Layer Perceptron
################ XOR Gate

import tensorflow as tf

# 학습 데이터 정의
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0], [1], [1], [0]]

# tf.keras.Sequential 모델 생성
model = tf.keras.Sequential()

# 모델에 레이어 추가: 1단
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)))

# 모델에 레이어 추가: 2단
model.add(tf.keras.layers.Dense(4, activation='relu'))

# 모델에 레이어 추가: 3단
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# 비용 최소화해주는 Optimizer
opti = tf.keras.optimizers.Adam(lr=0.01)

# 모델 준비: mean_squared_error
model.compile(loss='mse', optimizer=opti, metrics=['accuracy'])

# model.summary()
# 신경망 학습
model.fit(x_data, y_data, epochs=1000)



