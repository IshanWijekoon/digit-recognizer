import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Dataset Shapes")
print(f"x_train (Images for training): {x_train.shape}")
print(f"y_train (Labels for training): {y_train.shape}")
print(f"x_test (Images for testing):   {x_test.shape}")
print(f"y_test (Labels for testing):   {y_test.shape}")

plt.imshow(x_train[0], cmap='gray')
plt.title(f"The actual label is: {y_train[0]}")
plt.axis('off') 
plt.show()