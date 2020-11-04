########################### Binary Classification

# 필요한 패키지 임포트
import tensorflow as tf
import numpy as np

# 파일 로드 -> training data로 정의
xy = np.loadtxt('D:/DeepLearning/logistic/data_diabetes.csv', delimiter=',', dtype=np.float32)

# x,y 데이터 shape에 주의해 데이터 분할
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Sequential 모델 생성 및 레이어 추가
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(8, )),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='softmax')
])

# cost func, loss func 설정
# 모델 준비
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.005), metrics=['accuracy'])

# 모델 요약
model.summary()

# 모델 학습
history = model.fit(x_data, y_data, epochs=50)