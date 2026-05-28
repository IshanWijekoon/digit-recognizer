import tensorflow as tf

keras = tf.keras
layers = keras.layers
models = keras.models

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Flatten(input_shape = (28, 28)),

    layers.Dense(128, activation='relu'),

    layers.Dense(10, activation = 'softmax')
])

model.summary()

