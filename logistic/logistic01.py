######################### Binary Classification

import tensorflow as tf

# 학습 데이터
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

# Sequential 모델 생성 및 레이어 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2, )),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(units=2, activation='sigmoid')
])

# 모델 준비
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

# 모델 요약 정보 및 학습
model.summary()
history = model.fit(x_data, y_data, epochs=50)

# loss, accuracy 변화 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'r-', label='loss')
plt.title('###COST###')
plt.xlabel('epochs')
plt.ylabel('cost')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='accuracy')
plt.title('###ACCURACY###')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.show()