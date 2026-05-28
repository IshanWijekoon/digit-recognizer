import tensorflow as tf

keras = tf.keras
layers = keras.layers
models = keras.models

# load and normalize data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# neural network architecture
model = models.Sequential([
    layers.Flatten(input_shape = (28, 28)),

    layers.Dense(128, activation='relu'),

    layers.Dense(10, activation = 'softmax')
])

# compile the model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# trains the model
history = model.fit(x_train, y_train, epochs = 5, validation_split = 0.2)

# secret test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose = 2)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# saving the trained brain to a file so I can use it in our Pygame UI later.
model.save('mnist_model.keras')


