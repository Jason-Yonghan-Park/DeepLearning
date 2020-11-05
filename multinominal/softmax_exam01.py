############################### Multinominal Classification 실습

# 필요한 패키지 임포트
import tensorflow as tf
import numpy as np

# 데이터 읽어와서 training data 정의
animal = np.loadtxt('D:/DeepLearning/multinominal/data_zoo.csv', delimiter=',', dtype=np.float32)

# 데이터 슬라이싱
x_data = animal[:, 0:-1]
y_data = animal[:, -1]

# Sequential 이용해 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(16, )),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(units=7, activation='softmax')
])

# cost func, loss func 설정
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.005), metrics=['accuracy'])

# 모델 요약 정보
model.summary()

# 모델 학습
history = model.fit(x_data, y_data, epochs=50)

# cost, accuracy 시각화
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 폰트 지정 및 리소스 할당
fontLocation = 'C:/Windows/fonts/malgun.ttf'
fontName = font_manager.FontProperties(fname=fontLocation).get_name()
rc('font', family=fontName)

# 차트 서브플롯으로 출력
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'r-', label='loss')
plt.title('학습에 따른 cost 변화')
plt.xlabel('Epochs')
plt.ylabel('Cost')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.title('학습에 따른 accuracy 변화')
plt.xlabel('Epochs')
plt.ylabel('accuracy')

plt.show()

