###################### 텐서플로우 수학함수
import tensorflow as tf
import numpy as np

# 상수 정의
a = tf.constant(2)
b = tf.constant(5)

# 덧셈 뺄셈 텐서 형태로 출력
print(tf.add(a, b))
print(tf.divide(b, a))

# 제곱 및 거듭제곱 -> numpy 배열 형태로 출력
print(tf.square(a).numpy())
print(tf.pow(a,b).numpy())

# 텐서를 numpy로 변환 후 numpy 함수 사용
c = tf.add(a,b).numpy()
c_square = np.square(c, dtype=np.float32)

# numpy 배열 -> 텐서로 변환
c_tensor = tf.convert_to_tensor(c_square)
print(c_tensor)