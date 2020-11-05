################ Multinominal Classification
### softmax classifier

# 필요한 패키지 임포트
import tensorflow as tf

# 학습데이터 정의
x_data = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5], [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_data = [[2], [2], [2], [1], [1], [1], [0], [0]]

# Sequential 모델 생성 및 레이어 지정
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4, )),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

# cost func, loss func 설정
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.005), metrics=['accuracy'])

# 모델 요약 정보 출력
model.summary()

# 모델 학습
history = model.fit(x_data, y_data, epochs=50)

# cost, accuracy 시각화
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 폰트 지정
fontLocation = 'C:/Windows/fonts/malgun.ttf'
fontName = font_manager.FontProperties(fname=fontLocation).get_name()
rc("font", family=fontName)

# 차트 출력
plt.figure(figsize=(12, 4))
# 서브플롯 지정
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'r-', label='loss')
plt.title('학습에 따른 cost 변화')
plt.xlabel('epochs')
plt.ylabel('cost')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.title('학습에 따른 accuracy 변화')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.show()
