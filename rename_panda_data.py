import os

bild_ordner = r"C:\Users\Tim\Desktop\KI\labeling\Trainingsbilder\panda"

panda_index = 1

for datei in os.listdir(bild_ordner):
    alter_pfad = os.path.join(bild_ordner, datei)

    if datei.lower().endswith((".jpg", ".png", ".jpeg")):
       
        neuer_name = f"panda_{panda_index}.jpg"  
        panda_index += 1  # Zähler erhöhen
        

        neuer_pfad = os.path.join(bild_ordner, neuer_name)

        os.rename(alter_pfad, neuer_pfad)
        print(f"✅ {datei} → {neuer_name}")

