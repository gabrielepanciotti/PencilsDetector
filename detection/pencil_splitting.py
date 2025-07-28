"""
Module for detecting and splitting pencils of different colors.
"""
import csv
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import shutil
from image_utils.debug_manager import save_debug_image
from config import COLOR_RANGES


def calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calcola l'Intersection over Union (IoU) tra due bounding box.
    
    Args:
        box1: Primo bounding box in formato (x, y, w, h)
        box2: Secondo bounding box in formato (x, y, w, h)
        
    Returns:
        IoU tra i due bounding box (valore tra 0 e 1)
    """
    # Converti da (x, y, w, h) a (x1, y1, x2, y2)
    x1_1, y1_1, w1, h1 = box1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = box2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calcola le coordinate dell'intersezione
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calcola l'area dell'intersezione
    if x2_i <= x1_i or y2_i <= y1_i:
        # Non c'è intersezione
        return 0.0
    
    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calcola le aree dei due bounding box
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Calcola l'IoU
    iou = area_i / float(area1 + area2 - area_i)
    
    return iou


def detect_and_split_pencils(image: np.ndarray, color: str = 'green', debug: bool = False) -> List[Dict[str, Any]]:
    """
    Rileva e divide le matite del colore specificato nell'immagine.
    
    Args:
        image: Immagine originale in formato BGR
        color: Colore delle matite da rilevare ('green', 'red', ecc.)
        debug: Se True, salva le immagini di debug
        
    Returns:
        Lista di dizionari contenenti le informazioni sulle matite divise
    """
    # Converti in HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Crea la maschera per il colore specificato
    if color == 'red' or color == 'pink':
        # Per il rosso e il rosa dobbiamo combinare due intervalli
        mask1 = cv2.inRange(hsv, COLOR_RANGES[color]['lower1'], COLOR_RANGES[color]['upper1'])
        mask2 = cv2.inRange(hsv, COLOR_RANGES[color]['lower2'], COLOR_RANGES[color]['upper2'])
        color_mask = cv2.bitwise_or(mask1, mask2)
    elif color in COLOR_RANGES:
        color_mask = cv2.inRange(hsv, COLOR_RANGES[color]['lower'], COLOR_RANGES[color]['upper'])
    else:
        raise ValueError(f"Colore '{color}' non supportato. Colori disponibili: {list(COLOR_RANGES.keys())}")
    
    # Applica operazioni morfologiche per migliorare la maschera
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Salva solo la maschera finale se richiesto
    if debug:
        # Salviamo solo l'immagine della maschera del colore rilevato
        save_debug_image(color_mask, f"{color}_mask.jpg", f"{color}_detection")
    
    # Trova i contorni nella maschera
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra i contorni per area e rapporto altezza/larghezza
    # Imposta soglie di area minima diverse per colori diversi
    min_area = 40000   # Soglia più alta per il rosso, ma aumentiamo anche per gli altri colori
    filtered_contours = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
            
        # Verifica anche le proporzioni (le matite sono più alte che larghe)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(h) / w if w > 0 else 0
        
        # Le matite tendono ad essere più alte che larghe
        if aspect_ratio < 2.0:  # Deve essere almeno due volte più alta che larga
            continue
            
        # Aggiungiamo un filtro sulla larghezza minima solo per il rosso
        if w < 200:
            continue

        filtered_contours.append(c)
    
    # Lista per memorizzare tutte le matite del colore specificato (dopo la divisione)
    all_pencils = []
    
    # Debug: crea una cartella per questo colore e salva le informazioni richieste
    if debug:
        # Creiamo la directory per questo colore
        from image_utils.debug_manager import prepare_debug_directory
        debug_dir = prepare_debug_directory()
        color_debug_dir = os.path.join(debug_dir, color)
        # Rimuovi la cartella se esiste già
        if os.path.exists(color_debug_dir):
            shutil.rmtree(color_debug_dir)
        os.makedirs(color_debug_dir, exist_ok=True)
        
        # Salva l'immagine con i contorni filtrati
        contours_img = image.copy()
        cv2.drawContours(contours_img, filtered_contours, -1, (0, 255, 0), 2)
        filtered_contours_path = os.path.join(color_debug_dir, f"all_detected_{color}_pencils.jpg")
        cv2.imwrite(filtered_contours_path, contours_img)
        
        # Crea un file CSV per salvare le informazioni sulle matite rilevate
        csv_path = os.path.join(color_debug_dir, f"{color}_pencils_info.csv")
        csv_headers = ['id', 'x', 'y', 'width', 'height', 'area', 'aspect_ratio', 'num_divisions']
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_headers)
    
    # Per ogni contorno, crea un "pencil" e prova a dividerlo
    for i, contour in enumerate(filtered_contours):
        # Crea un bounding box
        x, y, w, h = cv2.boundingRect(contour)
        current_bbox = (x, y, w, h)
        
        # Verifica se questa matita si sovrappone significativamente con matite già rilevate
        is_duplicate = False
        for existing_pencil in all_pencils:
            existing_bbox = existing_pencil['bbox']
            iou = calculate_iou(current_bbox, existing_bbox)
            
            # Se l'IoU è alto, consideriamo questa come una duplicazione
            if iou > 0.2:  # Soglia di sovrapposizione abbassata per essere più restrittivi
                is_duplicate = True
                break
        
        # Se è una duplicazione, salta questo contorno
        if is_duplicate:
            continue
        
        # Estrai la regione della matita dall'immagine originale
        pencil_region = image[y:y+h, x:x+w]
        
        # Se debug è attivo, salva l'immagine dell'area rilevata
        if debug:
            area_path = os.path.join(color_debug_dir, f"{color}_area_{i+1}.jpg")
            cv2.imwrite(area_path, pencil_region)
        
        # Crea un dizionario pencil
        pencil = {
            'bbox': current_bbox,
            'contour': contour,
            'area': cv2.contourArea(contour),
            'color_name': color
        }
        
        # Prova a dividere la matita
        split_results = split_pencils(image, pencil, hsv_image=hsv, debug=debug, area_id=i+1)
        
        # Aggiorna il colore per ogni matita divisa
        for split_pencil in split_results:
            split_pencil['color_name'] = color
        
        # Verifica anche che le matite divise non si sovrappongano tra loro
        non_overlapping_results = []
        for j, split_pencil in enumerate(split_results):
            is_duplicate = False
            for k, existing_split in enumerate(non_overlapping_results):
                iou = calculate_iou(split_pencil['bbox'], existing_split['bbox'])
                if iou > 0.2:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                non_overlapping_results.append(split_pencil)
                
                # Se debug è attivo, aggiungi le informazioni al CSV
                if debug:
                    px, py, pw, ph = split_pencil['bbox']
                    area = cv2.contourArea(contour)
                    aspect_ratio = float(h) / w if w > 0 else 0
                    with open(os.path.join(color_debug_dir, f"{color}_pencils_info.csv"), 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([f"{i+1}_{j+1}", x + px, y + py, pw, ph, area, aspect_ratio, len(split_results)])
        
        # Aggiungi le matite divise non sovrapposte alla lista dei risultati
        all_pencils.extend(non_overlapping_results)
    
    return all_pencils


def split_pencils(image: np.ndarray, pencil: Dict[str, Any], hsv_image: np.ndarray = None, debug: bool = False, area_id: int = 0) -> List[Dict[str, Any]]:
    """
    Split a detected pencil that might actually be multiple pencils based on hue variations.
    Questa funzione generalizzata funziona con qualsiasi colore.
    Inoltre rileva la punta della matita e la unisce con la base.
    
    Args:
        image: Original image in BGR format
        pencil: Dictionary containing pencil information
        hsv_image: HSV version of the image (optional)
        debug: Whether to save debug images
        area_id: ID of the area for debug purposes
        
    Returns:
        List of dictionaries containing split pencil information
    """
    # Extract the pencil region
    x, y, w, h = pencil['bbox']
    pencil_region = image[y:y+h, x:x+w].copy()
    
    # Convert to HSV if not provided
    if hsv_image is None:
        hsv_region = cv2.cvtColor(pencil_region, cv2.COLOR_BGR2HSV)
    else:
        hsv_region = hsv_image[y:y+h, x:x+w].copy()
    
    # Get the color name from the pencil
    color_name = pencil.get('color_name', 'unknown')
    
    # Create a mask for the colored pixels in the region based on the pencil's color
    # Utilizziamo le definizioni di colore da config.py
    
    # Crea la maschera per il colore specificato
    if color_name == 'red' or color_name == 'pink':
        # Per il rosso e il rosa dobbiamo combinare due intervalli
        mask1 = cv2.inRange(hsv_region, COLOR_RANGES[color_name]['lower1'], COLOR_RANGES[color_name]['upper1'])
        mask2 = cv2.inRange(hsv_region, COLOR_RANGES[color_name]['lower2'], COLOR_RANGES[color_name]['upper2'])
        color_mask = cv2.bitwise_or(mask1, mask2)
    elif color_name in COLOR_RANGES:
        color_mask = cv2.inRange(hsv_region, COLOR_RANGES[color_name]['lower'], COLOR_RANGES[color_name]['upper'])
    else:
        # Se il colore non è supportato, usiamo una maschera che include tutti i pixel non neri
        # Questo è un fallback per evitare errori
        s_channel = hsv_region[:, :, 1]
        v_channel = hsv_region[:, :, 2]
        color_mask = cv2.bitwise_and(s_channel > 30, v_channel > 30)
        color_mask = color_mask.astype(np.uint8) * 255
    
    # Apply the mask to the HSV region
    hsv_colored = hsv_region.copy()
    hsv_colored[color_mask == 0] = 0
    
    # Extract only the hue channel for colored pixels
    hue_channel = hsv_colored[:, :, 0]
    
    # Apply median blur to reduce noise while preserving edges
    hue_blurred = cv2.medianBlur(hue_channel, 5)
    
    # Compute histogram of hue values to find dominant hues
    # Ignore zero values (non-colored pixels)
    hue_values = hue_blurred[hue_blurred > 0].flatten()
    
    # Save debug images if requested - versione ridotta
    if debug:
        # Creiamo una directory di debug specifica per ogni colore
        import os
        from image_utils.debug_manager import prepare_debug_directory
        debug_dir = prepare_debug_directory()
        
        # Salviamo solo l'immagine originale e la maschera
        save_debug_image(pencil_region, f"{color_name}_original_region.jpg", color_name)
        save_debug_image(color_mask, f"{color_name}_mask.jpg", color_name)
    
    if len(hue_values) == 0:
        # No colored pixels found
        return [pencil]
    
    # Use K-means clustering to find the dominant hue values
    # Convert to float32 for K-means
    hue_values = np.float32(hue_values.reshape(-1, 1))
    
    # Try to find 2 clusters (assuming 2 pencils)
    k = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(hue_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Sort centers by value
    centers = centers.flatten()
    centers_sorted = np.sort(centers)
    
    # Check if the hue difference is significant enough to indicate different pencils
    hue_diff = abs(centers_sorted[1] - centers_sorted[0])
    
    # Impostiamo soglie diverse per colori diversi
    if color_name == 'red':
        threshold = 25  # Soglia alta per il rosso
    elif color_name == 'black':
        threshold = 50  # Soglia molto alta per il nero per evitare divisioni errate
    else:
        threshold = 10  # Soglia standard per gli altri colori
        
    # Per il rosso, dobbiamo gestire il caso speciale del wrapping attorno a 180/0
    # Se un cluster è vicino a 0 e l'altro vicino a 180, la differenza reale è minore
    if color_name == 'red' and hue_diff > 150:  # Probabilmente un cluster è vicino a 0 e l'altro vicino a 180
        hue_diff = 180 - hue_diff  # La vera differenza è minore
    
    # Save debug info about clustering
    if debug:
        # Assicuriamoci che la directory esista
        from image_utils.debug_manager import prepare_debug_directory
        debug_dir = prepare_debug_directory()
        color_debug_dir = os.path.join(debug_dir, color_name)
        os.makedirs(color_debug_dir, exist_ok=True)
        
        # Create visualization of the clustering
        cluster_vis = pencil_region.copy()
        
        # Reconstruct the labels to match the original image shape
        labeled_image = np.zeros_like(hue_blurred)
        flat_mask = hue_blurred.flatten() > 0
        flat_labeled = np.zeros_like(hue_blurred.flatten())
        
        # Only assign labels where hue > 0
        idx = 0
        for i in range(len(flat_mask)):
            if flat_mask[i]:
                flat_labeled[i] = labels[idx][0]
                idx += 1
        
        labeled_image = flat_labeled.reshape(hue_blurred.shape)
        
        # Create a colored visualization
        for i in range(k):
            # Create a distinct color for this cluster
            if i == 0:
                color = (0, 0, 255)  # Red (in BGR)
            else:
                color = (0, 255, 0)  # Green (in BGR)
            
            # Apply color overlay
            overlay = np.zeros_like(cluster_vis)
            mask = (labeled_image == i).astype(np.uint8) * 255
            overlay[mask > 0] = color
            
            # Blend with original
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, cluster_vis, 1 - alpha, 0, cluster_vis)
        
        # Salva l'immagine con la visualizzazione del clustering
        split_vis_path = os.path.join(color_debug_dir, f"{color_name}_area_{area_id}_split_visualization.jpg")
        cv2.imwrite(split_vis_path, cluster_vis)
    
    if hue_diff < threshold:  # Soglia aumentata
        # Se la differenza di tonalità è troppo piccola, è probabilmente la stessa matita
        return [pencil]
    
    # Create masks for each hue cluster
    masks = []
    for i in range(k):
        # Create a mask for this cluster
        cluster_mask = np.zeros_like(hue_blurred, dtype=np.uint8)
        
        # Find pixels close to this cluster center
        lower_bound = centers[i] - max(3, hue_diff/2)
        upper_bound = centers[i] + max(3, hue_diff/2)
        
        # Set pixels within range to 255
        cluster_mask[(hue_blurred >= lower_bound) & (hue_blurred <= upper_bound) & (hue_blurred > 0)] = 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Per evitare che una singola matita venga divisa in più parti,
        # verifichiamo la connettività della maschera
        # Se ci sono più componenti connesse, teniamo solo la più grande
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cluster_mask, connectivity=8)
        
        if num_labels > 1:  # 0 è lo sfondo
            # Troviamo la componente connessa più grande (escludendo lo sfondo)
            max_area = 0
            max_label = 0
            for j in range(1, num_labels):  # Partiamo da 1 per saltare lo sfondo
                area = stats[j, cv2.CC_STAT_AREA]
                if area > max_area:
                    max_area = area
                    max_label = j
            
            # Creiamo una nuova maschera con solo la componente più grande
            if max_label > 0:
                new_mask = np.zeros_like(cluster_mask)
                new_mask[labels == max_label] = 255
                cluster_mask = new_mask
        
        masks.append(cluster_mask)
    
    # Find contours in each mask
    all_contours = []
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area - usiamo una soglia alta per eliminare le punte
        min_contour_area = 25000  # Aumentiamo la soglia base per tutti i colori
        
        # Filtriamo i contorni per area e anche per larghezza minima per il rosso
        filtered_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area <= min_contour_area:
                continue
                

            # Verifichiamo la larghezza solo per il rosso
            if color_name == 'red':
                x, y, w, h = cv2.boundingRect(c)
                if w < 70:  # Larghezza minima per le matite rosse
                    continue
                    
            filtered_contours.append(c)
        
        # Sort by area (largest first)
        filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)
        
        # Take the largest contour if any
        if filtered_contours:
            all_contours.append((i, filtered_contours[0]))
        
        # Save debug images if requested - versione ridotta
        if debug:
            # Disegniamo i contorni sulla visualizzazione del clustering
            cv2.drawContours(cluster_vis, filtered_contours, -1, (0, 255, 0), 2)
    
    # If we don't have at least 2 valid contours, return the original pencil
    if len(all_contours) < 2:
        return [pencil]
        
    # Se debug è attivo, salva un CSV con le informazioni sulle divisioni
    if debug and len(all_contours) >= 2:
        # Crea un dizionario con le informazioni sulle divisioni
        split_info = {
            'area_id': area_id,
            'color': color_name,
            'num_divisions': len(all_contours),
            'hue_diff': hue_diff,
            'threshold': threshold,
            'k_clusters': k
        }
        # Salva il CSV con le informazioni
        csv_path = os.path.join(color_debug_dir, f"{color_name}_area_{area_id}_split_info.csv")
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['key', 'value'])
            for key, value in split_info.items():
                writer.writerow([key, str(value)])
    
    # Create new pencils from the contours
    split_pencils = []
    
    # Non salviamo qui l'immagine di debug, lo facciamo già nella sezione precedente
    
    for i, (cluster_idx, contour) in enumerate(all_contours):
        # Get bounding box in the local coordinate system
        x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
        
        # Convert to global coordinate system
        global_x = x + x_cont
        global_y = y + y_cont
        
        # Se debug è attivo, salva l'immagine della divisione
        if debug:
            # Estrai la regione della matita divisa
            division_region = pencil_region[y_cont:y_cont+h_cont, x_cont:x_cont+w_cont].copy()
            # Salva l'immagine della divisione con un nome più descrittivo
            division_path = os.path.join(color_debug_dir, f"{color_name}_pencil_division_{area_id}_{i+1}.jpg")
            cv2.imwrite(division_path, division_region)
        
        # Create a new pencil dictionary
        new_pencil = pencil.copy()
        new_pencil['bbox'] = (global_x, global_y, w_cont, h_cont)
        new_pencil['contour'] = contour
        new_pencil['area'] = cv2.contourArea(contour)
        new_pencil['hue_value'] = centers[cluster_idx]  # Store the hue value for reference
        
        # Impostiamo il nome del colore uguale a quello della matita originale
        new_pencil['color_name'] = color_name
        
        split_pencils.append(new_pencil)
        
        # Non salviamo più l'immagine duplicata qui, poiché abbiamo già salvato l'immagine della divisione sopra
    
    # If we somehow didn't create any split pencils, return the original
    if not split_pencils:
        return [pencil]
        
    # Ora cerchiamo le punte delle matite
    pencils_with_tips = detect_and_merge_pencil_tips(image, split_pencils, hsv_image, debug, area_id, color_name)
    
    return pencils_with_tips


# La funzione split_green_pencils è stata rimossa poiché è stata sostituita dalla funzione generalizzata split_pencils


def detect_and_merge_pencil_tips(image: np.ndarray, pencils: List[Dict[str, Any]], hsv_image: np.ndarray = None, debug: bool = False, area_id: int = 0, color_name: str = 'unknown') -> List[Dict[str, Any]]:
    """
    Rileva le punte delle matite e le unisce con le basi corrispondenti.
    
    Args:
        image: Immagine originale in formato BGR
        pencils: Lista di dizionari contenenti le informazioni sulle matite (basi)
        hsv_image: Immagine HSV (opzionale)
        debug: Se True, salva le immagini di debug
        area_id: ID dell'area per scopi di debug
        color_name: Nome del colore delle matite
        
    Returns:
        Lista di dizionari contenenti le informazioni sulle matite con punte unite
    """
    # Importiamo COLOR_MAP da config
    from config import COLOR_MAP
    # Se non ci sono matite, restituisci la lista vuota
    if not pencils:
        return pencils
        
    # Converti in HSV se non fornito
    if hsv_image is None:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Crea la maschera per il colore specificato
    if color_name == 'red' or color_name == 'pink':
        # Per il rosso e il rosa dobbiamo combinare due intervalli
        mask1 = cv2.inRange(hsv_image, COLOR_RANGES[color_name]['lower1'], COLOR_RANGES[color_name]['upper1'])
        mask2 = cv2.inRange(hsv_image, COLOR_RANGES[color_name]['lower2'], COLOR_RANGES[color_name]['upper2'])
        color_mask = cv2.bitwise_or(mask1, mask2)
    elif color_name in COLOR_RANGES:
        color_mask = cv2.inRange(hsv_image, COLOR_RANGES[color_name]['lower'], COLOR_RANGES[color_name]['upper'])
    else:
        # Se il colore non è supportato, usiamo una maschera che include tutti i pixel non neri
        s_channel = hsv_image[:, :, 1]
        v_channel = hsv_image[:, :, 2]
        color_mask = cv2.bitwise_and(s_channel > 30, v_channel > 30)
        color_mask = color_mask.astype(np.uint8) * 255
    
    # Applica operazioni morfologiche per migliorare la maschera
    kernel = np.ones((3, 3), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Trova i contorni nella maschera
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtra i contorni per area massima (per trovare le punte)
    # Impostiamo soglie diverse per colori diversi
    if color_name == 'red' or color_name == 'pink':
        max_area = 40000  # Soglia più alta per rosso/rosa che tendono ad avere punte più grandi
        min_area = 100
    elif color_name == 'black':
        max_area = 25000  # Soglia specifica per il nero
        min_area = 50
    elif color_name == 'blue' or color_name == 'light_blue':
        max_area = 35000  # Soglia specifica per il blu/azzurro
        min_area = 70
    elif color_name == 'yellow':
        max_area = 30000  # Soglia specifica per il giallo
        min_area = 80
    elif color_name == 'purple':
        max_area = 28000  # Soglia specifica per il viola
        min_area = 60
    elif color_name == 'orange':
        max_area = 32000  # Soglia specifica per l'arancione
        min_area = 80
    else:
        max_area = 30000  # Soglia standard per gli altri colori
        min_area = 50     # Soglia minima ridotta per evitare di perdere punte piccole
    tip_contours = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            # Verifica anche le proporzioni (le punte sono di varie forme)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(h, w) / min(h, w) if min(h, w) > 0 else 1000
            
            # Usiamo criteri diversi in base al colore
            if color_name in ['red', 'pink']:
                max_aspect_ratio = 6.0  # Più permissivo per rosso/rosa
            elif color_name == 'black':
                max_aspect_ratio = 5.5  # Specifico per nero
            elif color_name in ['blue', 'light_blue']:
                max_aspect_ratio = 5.0  # Per blu/azzurro
            else:
                max_aspect_ratio = 4.5  # Standard per gli altri colori
            
            # Accettiamo solo le forme che rispettano il criterio di aspect ratio
            if aspect_ratio < max_aspect_ratio:
                tip_contours.append((x, y, w, h, c))
    
    # Debug: visualizza le punte rilevate
    if debug:
        # Assicuriamoci che la directory esista
        import os
        from image_utils.debug_manager import prepare_debug_directory
        debug_dir = prepare_debug_directory()
        color_debug_dir = os.path.join(debug_dir, color_name)
        os.makedirs(color_debug_dir, exist_ok=True)
        
        # Crea un'immagine con tutte le punte rilevate
        tips_img = image.copy()
        
        # Disegna un rettangolo per ogni punta rilevata
        for i, (x, y, w, h, c) in enumerate(tip_contours):
            cv2.rectangle(tips_img, (x, y), (x+w, y+h), (0, 255, 255), 2)  # Giallo per le punte
            # Aggiungi il testo con l'area e l'ID per debug
            area = cv2.contourArea(c)
            cv2.putText(tips_img, f"ID:{i} A:{int(area)}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Salva anche l'immagine di ogni singola punta rilevata
            # Estrai un'area leggermente più grande della punta
            border = 10
            x1 = max(0, x - border)
            y1 = max(0, y - border)
            x2 = min(image.shape[1], x + w + border)
            y2 = min(image.shape[0], y + h + border)
            
            # Estrai e salva l'immagine della punta singola
            tip_single_img = image[y1:y2, x1:x2].copy()
            tip_single_path = os.path.join(color_debug_dir, f"{color_name}_tip_{i}_solo.jpg")
            cv2.imwrite(tip_single_path, tip_single_img)
        
        # Salva l'immagine con tutte le punte rilevate
        tips_path = os.path.join(color_debug_dir, f"{color_name}_all_tips.jpg")
        cv2.imwrite(tips_path, tips_img)
        
        # Salva la maschera usata per rilevare le punte
        cv2.imwrite(os.path.join(color_debug_dir, f"{color_name}_tips_mask.jpg"), color_mask)
    
    # Lista per memorizzare le matite con le punte unite
    pencils_with_tips = []
    
    # Per ogni matita (base), cerca la punta corrispondente
    for i, pencil in enumerate(pencils):
        base_x, base_y, base_w, base_h = pencil['bbox']
        
        # Calcola il range x in cui cercare la punta - ampliamo il range in base al colore
        if color_name in ['red', 'pink']:
            x_min = base_x - base_w * 1.5  # Più margine per rosso/rosa
            x_max = base_x + base_w * 2.5
        elif color_name == 'black':
            x_min = base_x - base_w * 1.2  # Margine specifico per nero
            x_max = base_x + base_w * 2.2
        elif color_name in ['blue', 'light_blue', 'purple']:
            x_min = base_x - base_w * 1.3  # Margine per blu/azzurro/viola
            x_max = base_x + base_w * 2.3
        else:
            x_min = base_x - base_w  # Margine standard
            x_max = base_x + base_w * 2
        
        # Trova le punte che si sovrappongono con questo range x
        matching_tips = []
        for x, y, w, h, c in tip_contours:
            # Verifica se la punta si sovrappone con il range x della base
            # Consideriamo l'intero range della punta, non solo il centro
            tip_x_min = x
            tip_x_max = x + w
            
            # Verifichiamo se c'è sovrapposizione tra i range x
            if (tip_x_min <= x_max and tip_x_max >= x_min):
                # Verifica che la punta sia sopra o sotto la base (non al centro)
                # Usiamo criteri diversi in base al colore
                if color_name in ['red', 'pink']:
                    # Per rosso/rosa, permettiamo punte più vicine alla base
                    if y < base_y - h/4 or y > base_y + base_h - h/4:
                        matching_tips.append((x, y, w, h, c))
                elif color_name == 'black':
                    # Per nero, siamo più restrittivi
                    if y < base_y - h*0.6 or y > base_y + base_h - h*0.6:
                        matching_tips.append((x, y, w, h, c))
                else:
                    # Criterio standard per gli altri colori
                    if y < base_y - h/3 or y > base_y + base_h - h/3:
                        matching_tips.append((x, y, w, h, c))
        
        # Se non ci sono punte corrispondenti, mantieni la matita originale
        if not matching_tips:
            pencils_with_tips.append(pencil)
            continue
        
        # Trova la punta migliore in base alla posizione e dimensione
        best_tip = None
        best_score = float('inf')
        
        for x, y, w, h, c in matching_tips:
            # Calcola la distanza verticale dalla base
            if y < base_y:  # Punta sopra la base
                distance = base_y - (y + h)
            else:  # Punta sotto la base
                distance = y - (base_y + base_h)
                
            # Calcola la distanza orizzontale dal centro della base
            base_center_x = base_x + base_w / 2
            tip_center_x = x + w / 2
            horizontal_distance = abs(base_center_x - tip_center_x)
            
            # Calcola un punteggio che favorisce punte vicine e centrate
            # Parametri di scoring diversi per colore
            if color_name in ['red', 'pink']:
                # Per rosso/rosa, diamo più importanza alla vicinanza verticale
                score = distance * 1.5 + horizontal_distance * 1.5
                min_area_ratio = 0.03  # Più permissivo per rosso/rosa
            elif color_name == 'black':
                # Per nero, diamo più importanza all'allineamento orizzontale
                score = distance + horizontal_distance * 3
                min_area_ratio = 0.04  # Valore specifico per nero
            elif color_name in ['blue', 'light_blue']:
                # Per blu/azzurro, bilanciamo distanza verticale e orizzontale
                score = distance * 1.2 + horizontal_distance * 2
                min_area_ratio = 0.04
            else:
                # Criterio standard per gli altri colori
                score = distance + horizontal_distance * 2
                min_area_ratio = 0.05
            
            # Se la punta è molto più piccola della base, potrebbe essere rumore
            # Penalizziamo punte troppo piccole rispetto alla base
            area_ratio = cv2.contourArea(c) / pencil['area']
            if area_ratio < min_area_ratio:
                score += 1000  # Penalizza fortemente
                
            # Penalizza anche punte troppo grandi rispetto alla base
            if area_ratio > 0.8:
                score += 800  # Penalizza punte troppo grandi
            
            if score < best_score and distance >= 0:  # Assicuriamoci che non si sovrappongano
                best_score = score
                best_tip = (x, y, w, h, c)
        
        # Se abbiamo trovato una punta valida, uniscila con la base
        if best_tip:
            tip_x, tip_y, tip_w, tip_h, tip_contour = best_tip
            
            # Crea un nuovo bounding box che include sia la base che la punta
            new_x = min(base_x, tip_x)
            new_y = min(base_y, tip_y)
            new_w = max(base_x + base_w, tip_x + tip_w) - new_x
            new_h = max(base_y + base_h, tip_y + tip_h) - new_y
            
            # Crea una copia della matita originale con il nuovo bounding box
            new_pencil = pencil.copy()
            new_pencil['bbox'] = (new_x, new_y, new_w, new_h)
            new_pencil['has_tip'] = True
            new_pencil['tip_bbox'] = (tip_x, tip_y, tip_w, tip_h)
            
            # Debug: visualizza la matita con la punta unita
            if debug:
                # Crea un'immagine più grande per mostrare chiaramente la matita e la sua punta
                # Aggiungiamo un bordo per rendere l'immagine più grande e chiara
                border_size = 50
                region_x = max(0, new_x - border_size)
                region_y = max(0, new_y - border_size)
                region_w = min(image.shape[1] - region_x, new_w + border_size * 2)
                region_h = min(image.shape[0] - region_y, new_h + border_size * 2)
                
                # Estrai la regione allargata
                region_img = image[region_y:region_y+region_h, region_x:region_x+region_w].copy()
                
                # Adatta le coordinate al nuovo sistema di riferimento
                local_base_x = base_x - region_x
                local_base_y = base_y - region_y
                local_tip_x = tip_x - region_x
                local_tip_y = tip_y - region_y
                local_new_x = new_x - region_x
                local_new_y = new_y - region_y
                
                # Disegna i rettangoli
                cv2.rectangle(region_img, (local_base_x, local_base_y), 
                             (local_base_x+base_w, local_base_y+base_h), (0, 255, 0), 2)  # Verde per la base
                cv2.rectangle(region_img, (local_tip_x, local_tip_y), 
                             (local_tip_x+tip_w, local_tip_y+tip_h), (0, 255, 255), 2)  # Giallo per la punta
                cv2.rectangle(region_img, (local_new_x, local_new_y), 
                             (local_new_x+new_w, local_new_y+new_h), (255, 0, 0), 2)  # Blu per il box unito
                
                # Aggiungi etichette per chiarezza
                cv2.putText(region_img, "Base", (local_base_x, local_base_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(region_img, "Punta", (local_tip_x, local_tip_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(region_img, "Unito", (local_new_x, local_new_y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Salva l'immagine con la visualizzazione della fusione
                merged_path = os.path.join(color_debug_dir, f"{color_name}_pencil_{i+1}_merged.jpg")
                cv2.imwrite(merged_path, region_img)
                
                # Aggiungiamo questa matita all'immagine di riepilogo
                # Creiamo un dizionario per tenere traccia delle matite con punte
                if not hasattr(detect_and_merge_pencil_tips, 'summary_img'):
                    detect_and_merge_pencil_tips.summary_img = image.copy()
                
                # Disegna il rettangolo della matita con punta
                cv2.rectangle(detect_and_merge_pencil_tips.summary_img, (new_x, new_y), (new_x+new_w, new_y+new_h), 
                             COLOR_MAP.get(color_name, (0, 0, 255)), 3)  # Usa il colore della matita
                
                # Aggiungi etichetta con il colore e l'indice
                cv2.putText(detect_and_merge_pencil_tips.summary_img, f"{color_name} {i+1}", (new_x, new_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_MAP.get(color_name, (0, 0, 255)), 2)
                
                # Salva l'immagine di riepilogo aggiornata
                summary_path = os.path.join(debug_dir, f"all_pencils_with_tips.jpg")
                cv2.imwrite(summary_path, detect_and_merge_pencil_tips.summary_img)
                
                # Stampa a console per debug
                print(f"Rilevata punta per matita {color_name} {i+1}")
            
            pencils_with_tips.append(new_pencil)
        else:
            # Se non abbiamo trovato una punta valida, mantieni la matita originale
            pencils_with_tips.append(pencil)
    
    return pencils_with_tips
