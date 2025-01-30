import os 
import cv2
import numpy as np

bild_ordner = r"C:\Users\Tim\Desktop\KI\labeling\Trainingsbilder"
bilder = []
labels = []
class_mapping = {"koala" : 0, "Panda" : 1}

for datei in os.listdir(bild_ordner):
    if datei.endswith(".jpg") or datei.endswith(".png"):
        bild_pfad = os.path.join(bild_ordner, datei)

        bild = cv2.imread(bild_pfad, cv2.IMREAD_GRAYSCALE)
        bild = cv2.resize(bild, (64, 64))
        bild = bild / 255.0

        bilder.append(bild)
        labels.append(datei)

        tier_name = ''.join([char for char in datei if not char.isdigit()]).replace(".png", "").lower()
        if tier_name in class_mapping:
            label = class_mapping[tier_name]
        else:
            continue  

bilder = np.array(bilder)
bilder = bilder.reshape(-1, 64, 64, 1)
print(f"Geladene Bilder:  {bilder.shape}" )
print(bilder)
print(labels)