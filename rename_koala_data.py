import os

# 📂 Ordner mit den Bildern (Pfad anpassen!)
bild_ordner = r"C:\Users\Tim\Desktop\KI\labeling\Trainingsbilder\koala"

# 🔢 Zähler für Koala-Bilder
koala_index = 1

# Alle Dateien im Ordner durchgehen
for datei in os.listdir(bild_ordner):
    alter_pfad = os.path.join(bild_ordner, datei)

    # Prüfen, ob es eine Bilddatei ist
    if datei.lower().endswith((".jpg", ".png", ".jpeg")):
        # Prüfen, ob "gray" im Namen vorkommt → dann ist es ein Koala-Bild
        if "gray" in datei.lower():
            neuer_name = f"koala_{koala_index}.jpg"  # Neues Format (koala1.jpg, koala2.jpg, ...)
            koala_index += 1  # Zähler erhöhen
        else:
            continue  # Falls das Bild nicht umbenannt werden soll, überspringen

        neuer_pfad = os.path.join(bild_ordner, neuer_name)

        # ✅ Datei umbenennen
        os.rename(alter_pfad, neuer_pfad)
        print(f"✅ {datei} → {neuer_name}")

print("\n🎉 Alle Bilder wurden umbenannt!")
