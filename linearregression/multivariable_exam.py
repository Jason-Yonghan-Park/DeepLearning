###################### Multi Variable Linear Regression 실습

# 필요한 패키지 임포트
import tensorflow as tf
import numpy as np

# 파일 데이터 읽어옴
data = np.loadtxt(fname='D:/DeepLearning/linearregression/data_score.csv', delimiter=',', dtype=np.float32, skiprows=1)

# 데이터 분류 -> 배열 모든행의 0~2열이 입력데이터 / 3열이 결과 데이터
x_data = data[:, :-1]
y_data = data[:, [-1]]

# 데이터 체크
print("x_data shape: ", x_data.shape)
print("y_data shape: ", y_data.shape)

# Sequential 모델 생성
model = tf.keras.models.Sequential()

# 모델에 레이어 3개 추가
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(3, )))
model.add(tf.keras.layers.Dense(32))
model.add(tf.keras.layers.Dense(1))

# cost func, loss func 설정
# 모델 준비
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])

# 모델 학습
model.fit(x_data, y_data, epochs=100)

# 모델 새로운 데이터 예측
print(model.predict(np.array([[89,91,90], [71,75,79]])))