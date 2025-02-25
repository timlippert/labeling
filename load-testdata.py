# MNIST-Datensatz laden
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(train_images[i], cmap="gray")
    plt.title(f"Label: {train_labels[i]}")
    plt.axis("off")

plt.show()

print("Trainingsdatenform:", train_images.shape)  # z.B. (60000, 28, 28)
print("Testdatenform:", test_images.shape)        # z.B. (10000, 28, 28)
