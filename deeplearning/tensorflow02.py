############ 텐서플로우 기본 연산 -> 2.0 코드 사용
import tensorflow as tf

# 상수 정의
node1 = tf.constant(5.0, tf.float32)
node2 = tf.constant(7.0)
node3 = tf.add(node1, node2)

# TensorFlow 2.x 에서는 세션을 생성하지 않아도 노드에 실제 값이 적용
print(node1, "-", node3)
print(node3 + node1)