import keras
from keras import layers
import koala_training
import numpy as np
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Anzahl der Klassen im Datensatz bestimmen
num_classes = koala_training.y_train.shape[1]

# Kernel Regularizer definieren
kernel_regularizer = keras.regularizers.l2(0.001)



koala_training.X_train = koala_training.X_train.reshape(-1, 64, 64, 1)
koala_training.X_test = koala_training.X_test.reshape(-1, 64, 64, 1)
# keras.Sequential = Schichten, welche in der Liste stehen, werden im Modell der Reihe nach verbunden
model = keras.Sequential([

    # layers.Conv2D = convolutional layer, die für Bilder genutzt wird (32 -> Anzahl Filter/Kerne, (3,3) -> Größe der Filter (3x3 Pixel pro Filter)
    # activation = "relu" -> Aktivierungsfunktion, die negative Werte auf 0 setzt (ReLU = Rectified Linear Unit))
    # Zweck: Erlernt Merkmale aus Bildern wie Kanten oder Texturen
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(64, 64, 1)),
    
    # layers.MaxPooling2D((2,2)) = Verkleinert das Bild um den Faktor 2
    # reduziert die Anzahl der Berechnungen und hilft das Modell robuster gegen kleine Bildverschiebungen zu machen
    # Zweck: Verkleinert das Bild und behält nur die wichtigsten Informationen
    layers.MaxPooling2D((2,2)),
    
    # layers.Flatten = Schicht wandelt die 2D-Daten in eine 1D-Vektorform um
    # notwendig, um Daten für die Dense (vollverbundenen) Schichten vorzubereiten    
    # layers.Dense = vollverbundene Schicht mit 64 Neuronen (64 -> Anzahl der Neuronen, activation="relu" -> ReLU-Aktivierungsfunktion
    # für nicht-lineare Transformation)
    # Zweck: Schicht kombiniert die Merkmale aus vorherigen Schichten und lernt komplexe Muster
    layers.Dense(64, activation="relu", kernel_regularizer=kernel_regularizer),
    
    # layers.Dropout(0.5) -> 50% der Verbindungen werden pro Trainingsschritt zufällig deaktiviert
    # Zweck: Overfitting (Überanpassung an die Trainingsdaten) zu vermeiden
    layers.Dropout(0.3),
    
    # Erste Convolutional + Pooling-Schicht
    layers.Conv2D(32, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Zweite Convolutional + Pooling-Schicht
    layers.Conv2D(64, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Dritte Convolutional + Pooling-Schicht
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Vierte Convolutional + Pooling-Schicht
    layers.Conv2D(128, (3,3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Flatten + Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation="relu", kernel_regularizer=kernel_regularizer),
    layers.Dropout(0.6),
    layers.Dense(128, activation="relu", kernel_regularizer=kernel_regularizer),
    layers.Dropout(0.5),

    # Ausgangsschicht
    layers.Dense(num_classes, activation="softmax")
])

# Ausgabe der Modell-Zusammenfassung
model.summary()

# Kompilieren des Modells mit Adam-Optimizer und der korrekten Verlustfunktion
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",  
    metrics=["accuracy"]
)

# EarlyStopping Callback definieren
early_stopping = EarlyStopping(
    monitor="val_loss",  # Überwacht die Validierungsverlustfunktion
    patience=3,  # Stoppt nach 5 Epochen ohne Verbesserung
    restore_best_weights=True  # Stellt das beste Modell wieder her
)

# Modell trainieren und History speichern
history = model.fit(
    koala_training.X_train, 
    koala_training.y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(koala_training.X_test, koala_training.y_test),
    callbacks=[early_stopping],  # EarlyStopping Callback hinzufügen
    verbose=1
)

# Modell evaluieren
test_loss, test_acc = model.evaluate(koala_training.X_test, koala_training.y_test, batch_size=1)

# Trainings- und Validierungsverlust plotten
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# Modell speichern
model.save("koala_panda_model.keras")

# Testgenauigkeit ausgeben
print(f"Test accuracy: {test_acc:.4f}")