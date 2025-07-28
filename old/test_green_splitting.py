"""
Script di test per verificare la funzione split_pencils con matite verdi.
"""
import cv2
import numpy as np
import os
from detection.pencil_splitting import split_pencils
from image_utils.debug_manager import save_debug_image, DebugManager

# Inizializza il debug manager
debug_manager = DebugManager("debug")
# Crea una directory di debug separata per questo test
debug_dir = "debug/pencil_split_test"
os.makedirs(debug_dir, exist_ok=True)

# Funzione personalizzata per salvare le immagini di debug
def save_test_debug(image, filename):
    full_path = os.path.join(debug_dir, filename)
    cv2.imwrite(full_path, image)
    print(f"Salvata immagine di debug: {full_path}")

# Carica l'immagine
image_path = "data/pencils.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"Errore nel caricamento dell'immagine {image_path}")
    exit(1)

# Converti in HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Crea una maschera per i pixel verdi
# Utilizziamo valori più permissivi per il verde
lower_green = np.array([35, 30, 30])  # Ridotto saturation e value minimi
upper_green = np.array([90, 255, 255])  # Aumentato leggermente il range di hue
green_mask = cv2.inRange(hsv, lower_green, upper_green)

# Salva la maschera verde originale
save_test_debug(green_mask, "01_green_mask_original.jpg")

# Applica operazioni morfologiche per migliorare la maschera
kernel = np.ones((5, 5), np.uint8)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Trova i contorni nella maschera verde
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtra i contorni per area e rapporto altezza/larghezza
min_area = 5000  # Aumentata significativamente da 15000 a 30000
filtered_contours = []

for c in contours:
    area = cv2.contourArea(c)
    if area < min_area:
        continue
        
    # Verifica anche le proporzioni (le matite sono più alte che larghe)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(h) / w if w > 0 else 0
    
    # Le matite tendono ad essere più alte che larghe
    if aspect_ratio < 2.0:  # Aumentato da 3.0 a 5.0 - deve essere almeno cinque volte più alta che larga
        continue
        
    filtered_contours.append(c)

# Salva l'immagine con i contorni
contour_image = image.copy()
cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
save_test_debug(contour_image, "02_green_contours.jpg")

# Per ogni contorno, crea un "pencil" e prova a dividerlo
for i, contour in enumerate(filtered_contours):
    # Crea un bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crea un dizionario pencil
    pencil = {
        'bbox': (x, y, w, h),
        'contour': contour,
        'area': cv2.contourArea(contour),
        'color_name': 'green'
    }
    
    # Salva l'immagine della regione della matita
    pencil_region = image[y:y+h, x:x+w].copy()
    save_test_debug(pencil_region, f"03_pencil_region_{i}.jpg")
    
    print(f"Provo a dividere la matita {i} con bounding box {(x, y, w, h)}")
    
    # Prova a dividere la matita
    split_results = split_pencils(image, pencil, hsv_image=hsv, debug=True)
    
    print(f"La matita {i} è stata divisa in {len(split_results)} matite")
    
    # Visualizza i risultati dello split (prima dell'unione)
    split_image = image.copy()
    for j, split_pencil in enumerate(split_results):
        sx, sy, sw, sh = split_pencil['bbox']
        cv2.rectangle(split_image, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
        cv2.putText(split_image, f"Split {i}.{j}", (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Salva anche l'immagine di ogni singola parte divisa
        part_img = image[sy:sy+sh, sx:sx+sw].copy()
        save_test_debug(part_img, f"04_split_part_{i}_{j}.jpg")
    
    save_test_debug(split_image, f"05_split_result_{i}.jpg")
    
    print(f"La matita {i} è stata divisa in {len(split_results)} parti")

print("Test completato. Controlla la cartella debug/pencil_split_test per i risultati.")
