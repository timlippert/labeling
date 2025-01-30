import tensorflow as tf
import numpy as np

# MNIST-Datensatz laden
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Daten normalisieren (Werte auf [0,1] skalieren)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Beispiel: Zeige die ersten 5 Bilder als ASCII-Grafik mit ihrem Label
for i in range(5):
    print(f"\nBild {i+1}: Erwartete Zahl (Label): {train_labels[i]}\n")
    
    # Das Bild als ASCII-Grafik ausgeben
    for row in train_images[i]:
        print("".join("█" if pixel > 0.5 else " " for pixel in row))
    
    print("\n" + "-" * 28)  # Trennlinie für bessere Übersicht

# Form der Daten ausgeben
print("Trainingsdatenform:", train_images.shape)  # z.B. (60000, 28, 28)
print("Testdatenform:", test_images.shape)        # z.B. (10000, 28, 28)
