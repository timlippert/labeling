import os
import cv2
import numpy as np
import re
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import to_categorical




# Pfad zum Ordner mit den Trainingsbildern
bild_ordner = r"C:\Users\Tim\Desktop\KI\labeling\Trainingsbilder"

# Listen für Bilder & Labels
bilder = []
labels = []

# Klassen-Zuordnung (Nummern für jedes Tier)
class_mapping = {"koala": 0, "panda": 1}  

# Alle Dateien im Ordner durchgehen
for datei in os.listdir(bild_ordner):
    if datei.endswith((".jpg", ".png")):  # Falls es ein Bild ist
        bild_pfad = os.path.join(bild_ordner, datei)

        # Bild als Graustufen laden
        bild = cv2.imread(bild_pfad, cv2.IMREAD_GRAYSCALE)
        bild = cv2.resize(bild, (64, 64))  # Größe auf 64x64 ändern

        # Normalisierung der Pixelwerte (0 bis 1)
        bild = bild.astype("float32") / 255.0  

        # Tiername aus Dateiname extrahieren (z. B. "Koala1.png" → "koala")
        tier_name = re.match(r"[A-Za-z]+", os.path.splitext(datei)[0]).group(0).lower()

        # Überprüfen, ob das Tier in class_mapping ist
        if tier_name in class_mapping:
            label = class_mapping[tier_name]  # Zahl zuweisen (0 für Koala, 1 für Panda)
            bilder.append(bild)  # Bild speichern
            labels.append(label)  # Label speichern

# In NumPy-Arrays umwandeln
bilder = np.array(bilder).reshape(-1, 64, 64, 1)  # Für CNN: (Anzahl, 64, 64, 1)
labels = np.array(labels)

# One-Hot-Encoding der Labels
# One-Hot-Encoding  -> label wird zu einem Vektor, der überall 0 ist, außer an "seiner Stelle"
# -> "grün" = [1,0,0], "rot" = [0,1,0] "gelb" -> [0,0,1] => einfache Verarbeitung für Computer
labels = to_categorical(labels, num_classes=len(class_mapping))

# Train-Test-Split (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(bilder, labels, test_size=0.2, random_state=42)

# Infos ausgeben
print(f"Trainingsbilder: {X_train.shape}, Testbilder: {X_test.shape}")
print(f"Trainingslabels: {y_train.shape}, Testlabels: {y_test.shape}")
