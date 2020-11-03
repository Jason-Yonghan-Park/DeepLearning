######################## 텐서플로우 자료구조

### Tensor, Rank, Shape, Type

import tensorflow as tf

# 텐서플로우에서 난수 생성
# uniform() : 0~1 사이의 모든 수가 나올 확률이 동일한 균일 분포에서 난수 생성해주는 함수
# 첫번째 인수에 생성할 난수의 shape 지정 (갯수)
rand1 = tf.random.uniform([4], 0, 1)
rand2 = tf.random.normal([4], 0, 1)

# random 모듈 -> 난수 생성
import random
rand3 = random.random()

print("rand1: ", rand1);print("rand2: ", rand2);print("rand3: ", rand3)

# TensorFlow 변수 선성
# rank = 0 / Scalar, shape=[]
s = tf.Variable(483, tf.int32)
print('s: ', s.shape, '-', tf.rank(s))
# rank = 1 / Vector, shape=[3]
v = tf.Variable([1.1, 2.2, 3.3], tf.float32)
# rank = 2 / Matrix, shape=[3,3]
m = tf.Variable([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# rank = 3 / Cube, shape=[3,3,3]
c = tf.Variable([[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]])
print('c: ', c.shape, '-', tf.rank(c))



