import tensorflow as tf
import tensorflow_datasets as tfds

mnist_train = tfds.load("fashion_mnist", split="train")

assert isinstance(mnist_train, tf.data.Dataset)

print(type(mnist_train))

for item in mnist_train.take(1):
  print(type(item))
  print(item.keys())
  print(item['image'])
  print(item['label'])
