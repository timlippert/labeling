import keras
from keras import layers
import koala_training
import numpy as np
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Anzahl der Klassen im Datensatz bestimmen
num_classes = koala_training.y_train.shape[1]
#0.4516
#0.4785
#0.0968

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
    layers.Flatten(),
    
    # layers.Dense = vollverbundene Schicht mit 64 Neuronen (64 -> Anzahl der Neuronen, activation="relu" -> ReLU-Aktivierungsfunktion
    # für nicht-lineare Transformation)
    # Zweck: Schicht kombiniert die Merkmale aus vorherigen Schichten und lernt komplexe Muster
    layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    
    # layers.Dropout(0.5) -> 50% der Verbindungen werden pro Trainingsschritt zufällig deaktiviert
    # Zweck: Overfitting (Überanpassung an die Trainingsdaten) zu vermeiden
    layers.Dropout(0.3),
    
    # layers.Dense(num_classes, activation="softmax") = Ausgabeschicht mit Softmax-Aktivierung
    # Softmax gibt für jede Klasse eine Wahrscheinlichkeit aus
    # Zweck: Klassifikation des Bildes in eine der 'num_classes' Klassen
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

early_stopping = EarlyStopping(
    monitor="val_loss",  # Überwacht die Validierungsverlustfunktion
    patience=5,  # Stoppt nach 5 Epochen ohne Verbesserung
    restore_best_weights=True  # Stellt das beste Modell wieder her

)


test_loss, test_acc = model.evaluate(koala_training.X_test, koala_training.y_test, batch_size=1)
model.save("koala_panda_model.keras")  # Speichert das Modell




history = model.fit(
    koala_training.X_train, 
    koala_training.y_train, 
    epochs=25, 
    batch_size=32, 
    validation_data=(koala_training.X_test, koala_training.y_test),
    verbose=1
)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

model.save("koala_panda_model.keras")  # Speichert das Modell

print(f"Test accuracy: {test_acc:.4f}")

