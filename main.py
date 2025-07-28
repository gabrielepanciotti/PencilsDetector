"""
Main module for pencil detection and splitting of different colors.
"""
import os
import cv2
import numpy as np
import argparse
from typing import List, Dict, Any
from image_utils.loader import load_image
from image_utils.debug_manager import prepare_debug_directory, save_debug_image
from detection.pencil_splitting import detect_and_split_pencils


def main():
    """
    Funzione principale che esegue il rilevamento e la divisione delle matite di vari colori.
    """
    # Parsing degli argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Rileva e dividi le matite di diversi colori in un\'immagine.')
    parser.add_argument('--image', type=str, default='data/pencils.jpg', help='Percorso dell\'immagine da analizzare')
    parser.add_argument('--output', type=str, default='results', help='Directory di output per i risultati')
    parser.add_argument('--debug', action='store_true', help='Abilita la modalità debug per salvare immagini intermedie')
    parser.add_argument('--show', action='store_true', help='Mostra le immagini risultato')
    parser.add_argument('--colors', type=str, default='green,red,blue,yellow,purple,orange,pink,brown,light_blue,black', help='Colori da rilevare, separati da virgola (es. "green,red,blue")')
    args = parser.parse_args()
    
    # Prepara la directory di debug se necessario
    if args.debug:
        debug_dir = prepare_debug_directory()
        print(f"Debug mode enabled. Debug images will be saved to {debug_dir}")
    
    # Carica l'immagine
    image = load_image(args.image)
    print(f"Loaded image with shape: {image.shape}")
    
    # Ottieni la lista dei colori da rilevare
    colors = [color.strip().lower() for color in args.colors.split(',')]
    
    # Dizionario per memorizzare le matite rilevate per ogni colore
    all_pencils = {}
    result_image = image.copy()
    
    # Utilizziamo i colori BGR dal file di configurazione
    from config import COLOR_MAP
    
    # Rileva e dividi le matite per ogni colore
    for color in colors:
        try:
            pencils = detect_and_split_pencils(image, color=color, debug=args.debug)
            all_pencils[color] = pencils
            print(f"Rilevate {len(pencils)} matite {color}")
            
            # Visualizza i risultati per questo colore
            for i, pencil in enumerate(pencils):
                x, y, w, h = pencil['bbox']
                display_color = COLOR_MAP.get(color, (255, 255, 255))  # Bianco come fallback
                cv2.rectangle(result_image, (x, y), (x+w, y+h), display_color, 2)
                cv2.putText(result_image, f"{color.capitalize()} {i}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
        except ValueError as e:
            print(f"Errore nel rilevamento del colore {color}: {e}")
    
    # Crea la directory di output se non esiste
    os.makedirs(args.output, exist_ok=True)
    
    # Salva l'immagine risultato
    result_path = os.path.join(args.output, "pencils_detected.jpg")
    cv2.imwrite(result_path, result_image)
    print(f"\nRisultati salvati in {args.output}")
    print(f"Immagine risultato: {result_path}")
    
    # Mostra l'immagine se richiesto
    if args.show:
        cv2.imshow("Pencils Detected", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Esegui la funzione principale se il file viene eseguito direttamente
if __name__ == "__main__":
    main()
