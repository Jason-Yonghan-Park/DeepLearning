######################## 텐서플로우 기본 연산
##### tf 2.x -> 1.x 코드 사용하는 방법

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()


# TensorFlow 상수 정의 - 기본 그래프에 노드 추가
node1 = tf.constant(5.0, tf.float32)
node2 = tf.constant(7.0)
node3 = tf.add(node1, node2)

# 노드 정보만 출력 -> Tensor출력
print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)
