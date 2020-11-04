######################## 가중치(W)에 따른 비용 함수 cost(W) 그래프
import tensorflow as tf
import matplotlib.pyplot as plt

# 학습 데이터
X = [1., 2., 3.]
Y = [1., 2., 3.]

# 가중치(W) 및 비용(cost) 저장할 변수
W_val = []
cost_val = []

# 가중치(W) 비용(cost) 업데이트 저장
for i in range(-30, 50):
    # 가중치
    W = i * 0.1
    # Hypothesis
    hypothsis = tf.multiply(X, W)
    # Cost Func
    cost = tf.reduce_mean(tf.square(hypothsis - Y))

    W_val.append(W)
    cost_val.append(cost)

plt.plot(W_val, cost_val)
plt.show()