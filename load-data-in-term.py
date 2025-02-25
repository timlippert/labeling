import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

for i in range(5):
    print(f"\nBild {i+1}: Erwartete Zahl (Label): {train_labels[i]}\n")
    
    for row in train_images[i]:
        print("".join("█" if pixel > 0.5 else " " for pixel in row))
    
    print("\n" + "-" * 28)  

print("Trainingsdatenform:", train_images.shape)  # z.B. (60000, 28, 28)
print("Testdatenform:", test_images.shape)        # z.B. (10000, 28, 28)
