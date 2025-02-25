import matplotlib.pyplot as plt

import numpy as np

from scipy.ndimage import gaussian_filter

import cv2
 
def main():


    bild_pfad = r"C:\Users\Tim\Downloads\tzimas_converted.png"


    bild = plt.imread(bild_pfad)


    if len(bild.shape) == 3:  # Falls es ein Farbbild ist

        bild = cv2.cvtColor(bild, cv2.COLOR_RGB2GRAY)

    # Anwenden eines Gauss-Filters

    gefiltert = gaussian_filter(bild, sigma=2)

    # Sobel-Filter anwenden

    sobel_x = cv2.Sobel(gefiltert, cv2.CV_64F, 1, 0, ksize=3)  # Horizontaler Gradient

    sobel_y = cv2.Sobel(gefiltert, cv2.CV_64F, 0, 1, ksize=3)  # Vertikaler Gradient

    # Gradientmagnituden berechnen
    # helfen, Modellparameter anzupassen (durch "Anzeige", wo der steilste Abgang ist -> suchen Minimum)
    #  -> Länge des Gradienten zeigt an, wie steil der "Abgang" ist
    #  -> Ungewöhnlich hohe Magnituden lassen auf Vanishing Gradient (Gradienten werden sehr klein 
    #  -> Gewichte, vorallem in tieferen Schichten werden nur langsam oder garnicht aktualisiert)
    #  -> oder Exploding Gradient (Gewichte werden extrem groß, sodass Training unbrauchbar wird
    #     und das Modell instabil wird)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)


    # Normalisieren und in 8-Bit umwandeln (für Anzeige)

    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # Ergebnisse anzeigen

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)

    plt.imshow(bild, cmap='gray')

    plt.title('Originalbild')

    plt.axis('off')

    plt.subplot(1, 3, 2)

    plt.imshow(gefiltert, cmap='gray')

    plt.title('Gaussgefiltertes Bild')

    plt.axis('off')

    plt.subplot(1, 3, 3)

    plt.imshow(gradient_magnitude, cmap='gray')

    plt.title('Kanten mit Sobel')

    plt.axis('off')

    plt.show()
 

main()

 