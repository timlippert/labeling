import os

bild_ordner = r"C:\Users\Tim\Desktop\KI\labeling\Trainingsbilder\koala"

koala_index = 1

# Alle Dateien im Ordner durchgehen
for datei in os.listdir(bild_ordner):
    alter_pfad = os.path.join(bild_ordner, datei)

    if datei.lower().endswith((".jpg", ".png", ".jpeg")):
        if "gray" in datei.lower():
            neuer_name = f"koala_{koala_index}.jpg"  
            koala_index += 1  
        else:
            continue  

        neuer_pfad = os.path.join(bild_ordner, neuer_name)

        os.rename(alter_pfad, neuer_pfad)
        print(f"✅ {datei} → {neuer_name}")

