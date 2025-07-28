"""
Module for detecting and splitting pencils of different colors.
"""
import csv
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
import os
import shutil
from image_utils.debug_manager import save_debug_image
from config import COLOR_RANGES

# Evita errori di importazione circolare per i type hints
if TYPE_CHECKING:
    from pencil_splitting import PencilDetector


class PencilDetector:
    """
    Classe per la rilevazione e gestione delle matite.
    Centralizza le operazioni comuni per evitare duplicazioni di codice.
    """
    def __init__(self, image: np.ndarray, color_name: str = 'unknown', debug: bool = False):
        """
        Inizializza il rilevatore di matite.
        
        Args:
            image: Immagine originale in formato BGR
            color_name: Nome del colore delle matite da rilevare
            debug: Se True, salva le immagini di debug
        """
        self.image = image
        self.color_name = color_name
        self.debug = debug
        self.hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        self.color_mask = None
        self.debug_dir = None
        self.color_debug_dir = None
        
        if debug:
            from image_utils.debug_manager import prepare_debug_directory
            self.debug_dir = prepare_debug_directory()
            self.color_debug_dir = os.path.join(self.debug_dir, color_name)
            os.makedirs(self.color_debug_dir, exist_ok=True)
    
    def create_color_mask(self, hsv_image: np.ndarray = None) -> np.ndarray:
        """
        Crea una maschera per il colore specificato.
        Se la maschera è già stata calcolata per l'immagine HSV principale e non viene fornita
        un'immagine HSV specifica, riutilizza la maschera esistente.
        
        Args:
            hsv_image: Immagine HSV (opzionale, se non fornita usa self.hsv_image)
            
        Returns:
            Maschera binaria per il colore specificato
        """
        # Se non viene fornita un'immagine HSV specifica e abbiamo già calcolato la maschera
        # per l'immagine HSV principale, riutilizziamo quella
        if hsv_image is None:
            hsv_image = self.hsv_image
            if self.color_mask is not None:
                return self.color_mask
            
        # Crea la maschera per il colore specificato
        if self.color_name == 'red' or self.color_name == 'pink':
            # Per il rosso e il rosa dobbiamo combinare due intervalli
            mask1 = cv2.inRange(hsv_image, COLOR_RANGES[self.color_name]['lower1'], COLOR_RANGES[self.color_name]['upper1'])
            mask2 = cv2.inRange(hsv_image, COLOR_RANGES[self.color_name]['lower2'], COLOR_RANGES[self.color_name]['upper2'])
            color_mask = cv2.bitwise_or(mask1, mask2)
        elif self.color_name in COLOR_RANGES:
            color_mask = cv2.inRange(hsv_image, COLOR_RANGES[self.color_name]['lower'], COLOR_RANGES[self.color_name]['upper'])
        else:
            # Se il colore non è supportato, usiamo una maschera che include tutti i pixel non neri
            s_channel = hsv_image[:, :, 1]
            v_channel = hsv_image[:, :, 2]
            color_mask = cv2.bitwise_and(s_channel > 30, v_channel > 30)
            color_mask = color_mask.astype(np.uint8) * 255
        
        # Se stiamo usando l'immagine HSV principale, memorizziamo la maschera
        if hsv_image is self.hsv_image:
            self.color_mask = color_mask
            
        return color_mask
    
    def apply_morphology(self, mask: np.ndarray, kernel_size: int = 5, close_iterations: int = 2, open_iterations: int = 1) -> np.ndarray:
        """
        Applica operazioni morfologiche alla maschera.
        
        Args:
            mask: Maschera binaria
            kernel_size: Dimensione del kernel
            close_iterations: Numero di iterazioni per l'operazione di chiusura
            open_iterations: Numero di iterazioni per l'operazione di apertura
            
        Returns:
            Maschera binaria migliorata
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)
        return mask
    
    def find_contours(self, mask: np.ndarray) -> List:
        """
        Trova i contorni nella maschera.
        
        Args:
            mask: Maschera binaria
            
        Returns:
            Lista di contorni trovati
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Per i colori problematici, se non troviamo abbastanza contorni, proviamo un approccio alternativo
        if self.color_name in ['red', 'black', 'purple', 'pink'] and len(contours) < 5:
            # Proviamo un approccio basato sulla saturazione e valore piuttosto che sul colore
            # Questo è utile per rilevare le mine che hanno colori diversi dal corpo della matita
            h, s, v = cv2.split(self.hsv_image)
            
            # Creiamo una maschera basata principalmente su saturazione e valore
            # per catturare le mine che hanno colori diversi
            if self.color_name == 'red':
                # Per il rosso con mina marrone
                alt_mask = cv2.inRange(s, 20, 200) & cv2.inRange(v, 30, 200)
            elif self.color_name == 'black':
                # Per il nero con mina grigia
                alt_mask = cv2.inRange(s, 0, 80) & cv2.inRange(v, 30, 150)
            elif self.color_name == 'purple':
                # Per il viola
                alt_mask = cv2.inRange(s, 30, 200) & cv2.inRange(v, 30, 200)
            elif self.color_name == 'pink':
                # Per il rosa carne
                alt_mask = cv2.inRange(s, 20, 150) & cv2.inRange(v, 100, 230)
            
            # Applichiamo operazioni morfologiche per migliorare la maschera alternativa
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            alt_mask = cv2.morphologyEx(alt_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            alt_mask = cv2.morphologyEx(alt_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Salva la maschera alternativa se in modalità debug
            if self.debug:
                self.save_debug_image(alt_mask, f"{self.color_name}_alt_mask_for_tips.jpg")
            
            # Trova i contorni nella maschera alternativa
            alt_contours, _ = cv2.findContours(alt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Combina i contorni trovati con entrambi i metodi
            contours = contours + alt_contours
        
        return contours
    
    def filter_contours_by_area(self, contours: List, min_area: int, max_area: Optional[int] = None) -> List:
        """
        Filtra i contorni per area.
        
        Args:
            contours: Lista di contorni
            min_area: Area minima
            max_area: Area massima (opzionale)
            
        Returns:
            Lista di contorni filtrati
        """
        filtered_contours = []
        
        for c in contours:
            area = cv2.contourArea(c)
            if min_area <= area and (max_area is None or area <= max_area):
                filtered_contours.append(c)
                
        return filtered_contours
    
    def filter_contours_by_aspect_ratio(self, contours: List, min_ratio: float, max_ratio: float) -> List:
        """
        Filtra i contorni per rapporto d'aspetto.
        
        Args:
            contours: Lista di contorni
            min_ratio: Rapporto d'aspetto minimo (h/w)
            max_ratio: Rapporto d'aspetto massimo (h/w)
            
        Returns:
            Lista di contorni filtrati con bounding box
        """
        filtered_contours = []
        
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(h) / w if w > 0 else 1000
            
            if min_ratio <= aspect_ratio <= max_ratio:
                filtered_contours.append((x, y, w, h, c))
                
        return filtered_contours
    
    def get_min_max_area_for_tips(self) -> Tuple[int, int]:
        """
        Restituisce le soglie di area minima e massima per le punte in base al colore.
        
        Returns:
            Tupla (min_area, max_area)
        """
        if self.color_name == 'red':
            max_area = 50000  # Soglia molto più alta per il rosso che può avere punte di diversi colori
            min_area = 50     # Soglia minima molto ridotta per il rosso
        elif self.color_name == 'pink':
            max_area = 45000  # Soglia più alta per il rosa
            min_area = 70
        elif self.color_name == 'black':
            max_area = 30000  # Soglia aumentata per il nero
            min_area = 40     # Soglia minima ridotta per il nero
        elif self.color_name == 'blue' or self.color_name == 'light_blue':
            max_area = 35000  # Soglia specifica per il blu/azzurro
            min_area = 70
        elif self.color_name == 'yellow':
            max_area = 30000  # Soglia specifica per il giallo
            min_area = 80
        elif self.color_name == 'purple':
            max_area = 32000  # Soglia aumentata per il viola
            min_area = 50     # Soglia minima ridotta per il viola
        elif self.color_name == 'orange':
            max_area = 32000  # Soglia specifica per l'arancione
            min_area = 80
        elif self.color_name == 'green':
            max_area = 35000  # Soglia specifica per il verde
            min_area = 50     # Soglia minima ridotta per il verde
        else:
            max_area = 30000  # Soglia standard per gli altri colori
            min_area = 50     # Soglia minima ridotta per evitare di perdere punte piccole
            
        return min_area, max_area
    
    def get_max_aspect_ratio_for_tips(self) -> float:
        """
        Restituisce il rapporto d'aspetto massimo per le punte in base al colore.
        
        Returns:
            Rapporto d'aspetto massimo
        """
        if self.color_name == 'red':
            return 8.0  # Molto più permissivo per il rosso
        elif self.color_name == 'pink':
            return 7.0  # Più permissivo per il rosa
        elif self.color_name == 'black':
            return 6.5  # Più permissivo per il nero
        elif self.color_name in ['purple', 'violet']:
            return 6.0  # Più permissivo per il viola
        elif self.color_name in ['blue', 'light_blue']:
            return 5.5  # Per blu/azzurro
        elif self.color_name == 'green':
            return 5.0  # Specifico per il verde
        else:
            return 5.0  # Standard aumentato per gli altri colori
    
    def save_debug_image(self, image: np.ndarray, filename: str) -> str:
        """
        Salva un'immagine di debug.
        
        Args:
            image: Immagine da salvare
            filename: Nome del file
            
        Returns:
            Percorso completo del file salvato
        """
        if not self.debug:
            return ""
            
        filepath = os.path.join(self.color_debug_dir, filename)
        cv2.imwrite(filepath, image)
        return filepath


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
    # Crea un'istanza del rilevatore di matite
    detector = PencilDetector(image, color, debug)
    
    # Converti in HSV
    hsv = detector.hsv_image
    
    # Crea la maschera per il colore specificato
    color_mask = detector.create_color_mask(hsv)
    
    # Applica operazioni morfologiche per migliorare la maschera
    color_mask = detector.apply_morphology(color_mask)
    
    # Salva solo la maschera finale se richiesto
    if debug:
        # Salviamo solo l'immagine della maschera del colore rilevato
        detector.save_debug_image(color_mask, f"{color}_mask.jpg")
    
    # Trova i contorni nella maschera
    contours = detector.find_contours(color_mask)
    
    # Filtra i contorni per area e rapporto altezza/larghezza
    # Imposta soglie di area minima diverse per colori diversi
    min_area = 40000   # Soglia standard per la maggior parte dei colori
        
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
            
        # Aggiungiamo un filtro sulla larghezza minima
        if w < 200:
            continue

        filtered_contours.append(c)
    
    # Lista per memorizzare tutte le matite del colore specificato (dopo la divisione)
    all_pencils = []
    
    # Debug: crea una cartella per questo colore e salva le informazioni richieste
    if debug:
        # Creiamo la directory per questo colore
        debug_dir = detector.color_debug_dir
        # Rimuovi la cartella se esiste già
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        os.makedirs(debug_dir, exist_ok=True)
        
        # Salva l'immagine con i contorni filtrati
        contours_img = image.copy()
        cv2.drawContours(contours_img, filtered_contours, -1, (0, 255, 0), 2)
        filtered_contours_path = os.path.join(debug_dir, f"all_detected_{color}_pencils.jpg")
        cv2.imwrite(filtered_contours_path, contours_img)
        
        # Crea un file CSV per salvare le informazioni sulle matite rilevate
        csv_path = os.path.join(debug_dir, f"{color}_pencils_info.csv")
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
            if iou > 0.2: 
                is_duplicate = True
                break
        
        # Se è una duplicazione, salta questo contorno
        if is_duplicate:
            continue
        
        # Estrai la regione della matita dall'immagine originale
        pencil_region = image[y:y+h, x:x+w]
        
        # Se debug è attivo, salva l'immagine dell'area rilevata
        if debug:
            area_path = os.path.join(debug_dir, f"{color}_area_{i+1}.jpg")
            cv2.imwrite(area_path, pencil_region)
        
        # Crea un dizionario pencil
        pencil = {
            'bbox': current_bbox,
            'contour': contour,
            'area': cv2.contourArea(contour),
            'color_name': color
        }
        
        # Prova a dividere la matita
        split_results = split_pencils(image, pencil, detector, hsv_image=hsv, debug=debug, area_id=i+1)
        
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
                    with open(os.path.join(debug_dir, f"{color}_pencils_info.csv"), 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow([f"{i+1}_{j+1}", x + px, y + py, pw, ph, area, aspect_ratio, len(split_results)])
        
        # Aggiungi le matite divise non sovrapposte alla lista dei risultati
        all_pencils.extend(non_overlapping_results)
    
    # Rileva e unisci le punte delle matite
    all_pencils = detect_and_merge_pencil_tips(image, all_pencils, detector, hsv_image=hsv, debug=debug, color_name=color)
    
    return all_pencils


def split_pencils(image: np.ndarray, pencil: Dict[str, Any], detector: 'PencilDetector', hsv_image: np.ndarray = None, debug: bool = False, area_id: int = 0, force_clusters: int = None) -> List[Dict[str, Any]]:
    """
    Split a detected pencil that might actually be multiple pencils based on hue variations.
    Questa funzione generalizzata funziona con qualsiasi colore.
    Inoltre rileva la punta della matita e la unisce con la base.
    
    Args:
        image: Original image in BGR format
        pencil: Dictionary containing pencil information
        detector: Istanza di PencilDetector per operazioni di rilevamento
        hsv_image: HSV version of the image (optional)
        debug: Whether to save debug images
        area_id: ID of the area for debug purposes
        force_clusters: Force a specific number of clusters (optional)
        
    Returns:
        List of dictionaries containing split pencil information
    """
    # Get the color name from the pencil
    color_name = pencil.get('color_name', 'unknown')
    
    # Extract the pencil region
    x, y, w, h = pencil['bbox']
    pencil_region = image[y:y+h, x:x+w].copy()
    
    # Convert to HSV if not provided
    if hsv_image is None:
        hsv_region = cv2.cvtColor(pencil_region, cv2.COLOR_BGR2HSV)
    else:
        hsv_region = hsv_image[y:y+h, x:x+w].copy()
    
    # Create a mask for the colored pixels in the region
    # Utilizziamo il detector per creare la maschera
    color_mask = detector.create_color_mask(hsv_region)
    
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
        # Salviamo solo l'immagine originale e la maschera
        detector.save_debug_image(pencil_region, f"{color_name}_original_region.jpg")
        detector.save_debug_image(color_mask, f"{color_name}_mask.jpg")
    
    if len(hue_values) == 0:
        # No colored pixels found
        return [pencil]
    
    # Use K-means clustering to find the dominant hue values
    # Convert to float32 for K-means
    hue_values = np.float32(hue_values.reshape(-1, 1))
    
    # Determina il numero di cluster da utilizzare
    # Se force_clusters è specificato, usa quel valore
    # Altrimenti usa 2 come default (per dividere una matita in due)
    if force_clusters is not None:
        k = force_clusters
    else:
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
        detector.save_debug_image(cluster_vis, f"{color_name}_area_{area_id}_split_visualization.jpg")
    
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
        cluster_mask = detector.apply_morphology(cluster_mask, kernel_size=5, close_iterations=2, open_iterations=1)
        
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
        contours = detector.find_contours(mask)
        
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
        csv_path = os.path.join(detector.color_debug_dir, f"{color_name}_area_{area_id}_split_info.csv")
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
            detector.save_debug_image(division_region, f"{color_name}_pencil_division_{area_id}_{i+1}.jpg")
        
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
    #pencils_with_tips = detect_and_merge_pencil_tips(image, split_pencils, detector, hsv_image=hsv_image, debug=debug, area_id=area_id, color_name=color_name)
    
    return split_pencils

def detect_and_merge_pencil_tips(image: np.ndarray, pencils: List[Dict[str, Any]], detector: 'PencilDetector', hsv_image: np.ndarray = None, debug: bool = False, area_id: int = 0, color_name: str = 'unknown') -> List[Dict[str, Any]]:
    # Definiamo una funzione interna per salvare immagini di debug CSV
    def save_debug_csv(filepath, data):
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['key', 'value'])
            for key, value in data.items():
                writer.writerow([key, str(value)])
    """
    Rileva le punte delle matite e le unisce con le basi corrispondenti.
    
    Args:
        image: Immagine originale
        pencils: Lista di matite rilevate
        detector: Rilevatore di matite
        hsv_image: Immagine HSV (opzionale)
        debug: Se True, salva immagini di debug
        area_id: ID dell'area per scopi di debug
        color_name: Nome del colore delle matite
        
    Returns:
        Lista di dizionari contenenti le informazioni sulle matite con punte unite
    """
    # Se non ci sono matite, restituisci la lista vuota
    if not pencils:
        return pencils
    
    # Utilizziamo il detector fornito
        
    # Converti in HSV se non fornito
    if hsv_image is None:
        hsv_image = detector.hsv_image
    
    # Crea la maschera per il colore specificato
    color_mask = detector.create_color_mask(hsv_image)
    
    # Solo per il rilevamento delle punte, aggiungiamo maschere speciali per colori problematici
    # Questo non influisce sul rilevamento delle basi delle matite
    if color_name == 'red':
        # Per il rosso, includiamo anche tonalità marroni per catturare le mine marroni
        brown_lower = np.array([5, 30, 30])
        brown_upper = np.array([20, 200, 200])
        brown_mask = cv2.inRange(hsv_image, brown_lower, brown_upper)
        # Creiamo una maschera combinata solo per le punte
        tip_mask = cv2.bitwise_or(color_mask, brown_mask)
        # Applichiamo operazioni morfologiche per migliorare la maschera
        tip_mask = detector.apply_morphology(tip_mask, kernel_size=3, close_iterations=1, open_iterations=1)
        # Salva la maschera combinata se in modalità debug
        if debug:
            detector.save_debug_image(tip_mask, f"{color_name}_tip_mask_with_brown.jpg")
    elif color_name == 'black':
        # Per il nero, includiamo anche grigi più chiari per catturare le mine grigie
        gray_lower = np.array([0, 0, 30])
        gray_upper = np.array([180, 50, 150])
        gray_mask = cv2.inRange(hsv_image, gray_lower, gray_upper)
        # Creiamo una maschera combinata solo per le punte
        tip_mask = cv2.bitwise_or(color_mask, gray_mask)
        # Applichiamo operazioni morfologiche per migliorare la maschera
        tip_mask = detector.apply_morphology(tip_mask, kernel_size=3, close_iterations=1, open_iterations=1)
        # Salva la maschera combinata se in modalità debug
        if debug:
            detector.save_debug_image(tip_mask, f"{color_name}_tip_mask_with_gray.jpg")
    elif color_name == 'purple' or color_name == 'violet':
        # Per il viola, includiamo una gamma molto più ampia di saturazione e valore
        # Usiamo valori di saturazione e valore molto più bassi per catturare anche le mine più chiare
        purple_lower1 = np.array([120, 5, 5])  # Valori molto più permissivi
        purple_upper1 = np.array([170, 255, 255])
        purple_mask1 = cv2.inRange(hsv_image, purple_lower1, purple_upper1)
        
        # Aggiungiamo anche una maschera per catturare grigi/neri che potrebbero essere parte della mina
        gray_lower = np.array([0, 0, 5])
        gray_upper = np.array([180, 30, 150])
        gray_mask = cv2.inRange(hsv_image, gray_lower, gray_upper)
        
        # Aggiungiamo una maschera per catturare blu scuro che potrebbero essere confusi con viola
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([115, 255, 150])
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)
        
        # Creiamo una maschera combinata solo per le punte
        tip_mask = cv2.bitwise_or(color_mask, purple_mask1)
        tip_mask = cv2.bitwise_or(tip_mask, gray_mask)
        tip_mask = cv2.bitwise_or(tip_mask, blue_mask)
        
        # Applichiamo operazioni morfologiche per migliorare la maschera
        # Usiamo operazioni più aggressive per connettere aree frammentate
        tip_mask = detector.apply_morphology(tip_mask, kernel_size=5, close_iterations=3, open_iterations=1)
        
        # Salva la maschera combinata se in modalità debug
        if debug:
            detector.save_debug_image(tip_mask, f"{color_name}_tip_mask_expanded.jpg")
    elif color_name == 'pink':
        # Per il rosa, creiamo una maschera più precisa per evitare falsi positivi con il rosso
        # Usiamo un range più stretto per il rosa tradizionale
        pink_lower1 = np.array([161, 40, 100])
        pink_upper1 = np.array([170, 150, 255])
        # E un range più stretto per il rosa carne
        pink_lower2 = np.array([5, 50, 150])
        pink_upper2 = np.array([15, 150, 255])
        
        pink_mask1 = cv2.inRange(hsv_image, pink_lower1, pink_upper1)
        pink_mask2 = cv2.inRange(hsv_image, pink_lower2, pink_upper2)
        
        # Combiniamo le due maschere
        tip_mask = cv2.bitwise_or(pink_mask1, pink_mask2)
        # Applichiamo operazioni morfologiche per migliorare la maschera
        tip_mask = detector.apply_morphology(tip_mask, kernel_size=3, close_iterations=1, open_iterations=1)
        # Salva la maschera combinata se in modalità debug
        if debug:
            detector.save_debug_image(tip_mask, f"{color_name}_tip_mask_refined.jpg")
    else:
        # Per gli altri colori, usiamo la maschera standard
        tip_mask = color_mask.copy()
    
    # Applica operazioni morfologiche per migliorare la maschera delle basi
    color_mask = detector.apply_morphology(color_mask, kernel_size=3, close_iterations=1, open_iterations=1)
    
    # Trova i contorni nella maschera speciale per le punte
    contours = detector.find_contours(tip_mask)
    
    # Filtra i contorni per area massima (per trovare le punte)
    # Impostiamo soglie diverse per colori diversi
    min_area, max_area = detector.get_min_max_area_for_tips()
    
    # Per i colori problematici, utilizziamo soglie di area ancora più permissive
    if color_name == 'red':
        min_area = min_area // 2  # Dimezziamo la soglia minima per il rosso
        max_area = max_area * 2  # Raddoppiamo la soglia massima
    elif color_name in ['black', 'purple', 'pink']:
        min_area = min_area // 1.5  # Riduciamo la soglia minima
        max_area = int(max_area * 1.5)  # Aumentiamo la soglia massima
    
    # Filtriamo i contorni per area e rapporto d'aspetto
    tip_contours = []
    max_aspect_ratio = detector.get_max_aspect_ratio_for_tips()
    
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            # Verifica anche le proporzioni (le punte sono di varie forme)
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = max(h, w) / min(h, w) if min(h, w) > 0 else 1000
            
            # Accettiamo solo le forme che rispettano il criterio di aspect ratio
            if aspect_ratio < max_aspect_ratio:
                tip_contours.append((x, y, w, h, c))
    
    # Salva un'immagine di debug con i contorni delle punte
    if debug:
        debug_image = image.copy()
        cv2.drawContours(debug_image, [c for _, _, _, _, c in tip_contours], -1, (0, 255, 0), 2)
        detector.save_debug_image(debug_image, f"{color_name}_tip_contours.jpg")
        
        # Non salviamo più tutte le punte rilevate, solo quelle scelte
    
    # Crea una copia della lista di matite per aggiungere le punte
    pencils_with_tips = pencils.copy()
    
    # Per ogni matita (base), cerca la punta corrispondente
    for i, pencil in enumerate(pencils):
        base_x, base_y, base_w, base_h = pencil['bbox']
        
        # Calcola il punto centrale in alto della base
        base_center_x = base_x + base_w // 2
        base_top_y = base_y
        
        # Definiamo il margine orizzontale in base al colore
        if color_name in ['red', 'pink']:
            h_margin = base_w // 1.2  # Margine ancora più ampio per rosso/rosa
        elif color_name == 'black':
            h_margin = base_w // 1.5  # Margine più ampio per nero
        elif color_name in ['purple', 'violet']:
            h_margin = base_w // 1.4  # Margine più ampio per viola
        else:
            h_margin = base_w // 2  # Margine standard per gli altri colori
        
        # Inizializza le variabili per la migliore punta
        best_tip = None
        best_score = float('inf')
        
        # Cerca tra tutte le punte rilevate
        for j, (tip_x, tip_y, tip_w, tip_h, tip_contour) in enumerate(tip_contours):
            # Calcola il punto centrale in basso della punta
            tip_center_x = tip_x + tip_w // 2
            tip_bottom_y = tip_y + tip_h
            
            # Verifica se la punta è sopra la base (con una tolleranza)
            # Usiamo tolleranze diverse in base al colore
            if color_name in ['red', 'pink']:
                v_tolerance = 70  # Tolleranza molto maggiore per rosso/rosa
            elif color_name == 'black':
                v_tolerance = 60  # Tolleranza maggiore per nero
            elif color_name in ['purple', 'violet']:
                v_tolerance = 150  # Tolleranza ESTREMAMENTE maggiore per viola
            else:
                v_tolerance = 40  # Tolleranza aumentata per tutti i colori
                
            if tip_bottom_y > base_top_y - v_tolerance:
                continue  # La punta non è sopra la base
            
            # Verifica se la punta è orizzontalmente allineata con la base
            if abs(tip_center_x - base_center_x) > h_margin:
                continue  # La punta non è allineata orizzontalmente con la base
            
            # Calcola un punteggio per questa punta
            # Il punteggio è una combinazione di:
            # - Distanza verticale (più piccola è meglio)
            # - Distanza orizzontale dal centro (più piccola è meglio)
            # - Differenza di dimensione (penalizziamo punte troppo piccole o troppo grandi rispetto alla base)
            
            # Usiamo pesi diversi in base al colore
            if color_name == 'red':
                v_weight = 0.5  # Peso molto minore per la distanza verticale per il rosso
                h_weight = 3.0  # Peso molto maggiore per l'allineamento orizzontale
            elif color_name == 'pink':
                v_weight = 0.7  # Peso minore per la distanza verticale per il rosa
                h_weight = 2.5  # Peso maggiore per l'allineamento orizzontale
            elif color_name == 'black':
                v_weight = 0.8  # Peso minore per il nero
                h_weight = 2.5  # Peso maggiore per il nero
            elif color_name == 'purple' or color_name == 'violet':
                v_weight = 0.2  # Peso MOLTO minore per il viola
                h_weight = 1.0  # Peso ridotto per l'allineamento orizzontale
            else:
                v_weight = 1.2  # Peso standard per la distanza verticale
                h_weight = 1.8  # Peso standard per l'allineamento orizzontale
            
            v_distance = base_top_y - tip_bottom_y
            h_distance = abs(tip_center_x - base_center_x)
            
            # Penalizza le punte troppo piccole o troppo grandi rispetto alla base
            tip_area = cv2.contourArea(tip_contour)
            base_area = pencil['area']
            size_ratio = tip_area / base_area if base_area > 0 else 1000
            
            # Definiamo range accettabili per il rapporto di dimensione
            if color_name == 'red':
                min_size_ratio = 0.001  # Estremamente permissivo per il rosso
                max_size_ratio = 0.5    # Limite superiore aumentato
            elif color_name == 'pink':
                min_size_ratio = 0.005  # Molto permissivo per il rosa
                max_size_ratio = 0.45
            elif color_name == 'black':
                min_size_ratio = 0.003  # Più permissivo per il nero
                max_size_ratio = 0.4
            elif color_name == 'purple' or color_name == 'violet':
                min_size_ratio = 0.0001  # Estremamente permissivo per il viola
                max_size_ratio = 0.6     # Limite superiore molto aumentato
            elif color_name == 'green':
                min_size_ratio = 0.008  # Specifico per il verde
                max_size_ratio = 0.4
            else:
                min_size_ratio = 0.008  # Standard aumentato
                max_size_ratio = 0.38
                
            # Penalità per dimensioni fuori range
            if size_ratio < min_size_ratio or size_ratio > max_size_ratio:
                size_penalty = 1000  # Penalità alta per dimensioni fuori range
            else:
                size_penalty = 0
            
            # Calcola il punteggio finale
            score = v_weight * v_distance + h_weight * h_distance + size_penalty
            
            # Se questo punteggio è migliore del precedente, aggiorna
            if score < best_score:
                best_score = score
                best_tip = (j, tip_x, tip_y, tip_w, tip_h, tip_contour)
        
        # Se abbiamo trovato una punta valida, aggiorna la matita
        if best_tip:
            j, tip_x, tip_y, tip_w, tip_h, tip_contour = best_tip
            
            # Crea un nuovo bounding box che include sia la base che la punta
            new_x = min(base_x, tip_x)
            new_y = min(base_y, tip_y)
            new_w = max(base_x + base_w, tip_x + tip_w) - new_x
            new_h = max(base_y + base_h, tip_y + tip_h) - new_y
            
            # Aggiorna il bounding box della matita
            pencils_with_tips[i]['bbox'] = (new_x, new_y, new_w, new_h)
            
            # Aggiorna anche l'area della matita
            # Nota: non aggiorniamo il contorno perché sarebbe complicato unire i due contorni
            pencils_with_tips[i]['area'] = pencils_with_tips[i]['area'] + cv2.contourArea(tip_contour)
            
            # Aggiungi informazioni sulla punta per debug
            pencils_with_tips[i]['has_tip'] = True
            pencils_with_tips[i]['tip_bbox'] = (tip_x, tip_y, tip_w, tip_h)
            
            # Stampa un messaggio di debug
            if debug:
                print(f"Punta {j} (area: {cv2.contourArea(tip_contour)}) unita alla matita {i} (colore: {color_name})")
                
                # Salva un'immagine della matita con la punta
                merged_img = image.copy()
                # Disegna la base in blu
                cv2.rectangle(merged_img, (base_x, base_y), (base_x + base_w, base_y + base_h), (255, 0, 0), 2)
                # Disegna la punta scelta in verde
                cv2.rectangle(merged_img, (tip_x, tip_y), (tip_x + tip_w, tip_y + tip_h), (0, 255, 0), 2)
                # Disegna una linea che collega la base alla punta
                cv2.line(merged_img, (base_center_x, base_top_y), (tip_center_x, tip_bottom_y), (0, 255, 255), 2)
                # Aggiungi informazioni sul punteggio
                cv2.putText(merged_img, f"Score: {best_score:.2f}", (tip_x, tip_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                # Salva l'immagine
                detector.save_debug_image(merged_img, f"{color_name}_pencil_{i}_with_chosen_tip.jpg")
                
                # Salva anche l'immagine della sola punta scelta
                tip_img = image[tip_y:tip_y+tip_h, tip_x:tip_x+tip_w].copy()
                detector.save_debug_image(tip_img, f"{color_name}_chosen_tip_{i}.jpg")
        else:
            # Se non abbiamo trovato una punta, aggiungi questa informazione
            pencils_with_tips[i]['has_tip'] = False
    
    # Debug: salva un'immagine con tutte le matite e le punte
    if debug:
        merged_img = image.copy()
        
        # Disegna tutte le matite
        for i, pencil in enumerate(pencils_with_tips):
            x, y, w, h = pencil['bbox']
            # Colore diverso in base a se ha una punta o meno
            if pencil.get('has_tip', False):
                color = (0, 255, 0)  # Verde per le matite con punta
                # Disegna anche la punta
                tip_x, tip_y, tip_w, tip_h = pencil['tip_bbox']
                cv2.rectangle(merged_img, (tip_x, tip_y), (tip_x+tip_w, tip_y+tip_h), (0, 255, 255), 2)
            else:
                color = (0, 0, 255)  # Rosso per le matite senza punta
            
            cv2.rectangle(merged_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(merged_img, f"ID: {i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Salva l'immagine
        detector.save_debug_image(merged_img, f"{color_name}_all_merged_pencils.jpg")
    
    return pencils_with_tips
