# FlyCatcher - Sistema Avanzato di Rilevamento Matite Colorate

Un sistema di computer vision avanzato per rilevare, contare e localizzare matite colorate nelle immagini, con particolare attenzione al rilevamento accurato delle punte e alla gestione di matite sovrapposte o vicine.

## Caratteristiche Principali

- **Rilevamento Multi-Colore**: Supporto per matite di diversi colori (verde, rosso, blu, giallo, viola, arancione, rosa, marrone, azzurro, nero)
- **Divisione Intelligente**: Algoritmo di clustering K-means per separare matite dello stesso colore vicine o sovrapposte
- **Rilevamento Punte**: Sistema specializzato per identificare le punte delle matite e associarle correttamente alle basi
- **Gestione Casi Speciali**: Trattamento specifico per colori problematici come rosso (gestione del wrapping hue), viola (maschere ampliate) e rosa (filtri aggiuntivi)
- **Visualizzazione Avanzata**: Output visivo con bounding box colorate e etichette per ogni matita rilevata
- **Modalità Debug**: Generazione di immagini intermedie per analizzare ogni fase del processo di rilevamento

## Architettura del Sistema

```
FlyCatcher/
├── main.py                  # Script principale per l'esecuzione del rilevamento
├── config.py                # Configurazioni (range HSV, mappature colori)
├── detection/
│   ├── pencil_splitting.py  # Algoritmi di divisione e rilevamento punte
│   └── pencil_detector.py   # Classe base per il rilevamento delle matite
├── results/                 # Directory per i risultati
├── debug/                   # Directory per le immagini di debug
└── README.md                # Documentazione
│   ├── preprocessing.py # Preprocessing, blur, HSV
│   └── visualization.py # Annotazioni e salvataggi
│
├── detection/          # Moduli di rilevamento
│   ├── __init__.py
│   ├── color_masks.py  # Creazione maschere colore (HSV)
│   ├── contour_utils.py # Funzioni per trovare bounding box
│   └── classification.py # Dominant color, color classification
│
├── results/            # Gestione dei risultati
│   ├── __init__.py
│   ├── exporter.py     # Esportazione JSON
│   └── summary.py      # Stampe di riepilogo
│
└── data/               # Cartella per le immagini di input
```

## Requisiti

- Python 3.6+
- OpenCV
- NumPy

## Installazione

```bash
pip install -r requirements.txt
```

## Utilizzo

```bash
python -m pencil_detector.run --image data/pencils.jpg --output-dir output
```

### Opzioni

- `--image`: Percorso dell'immagine di input (obbligatorio)
- `--output-dir`: Directory dove salvare i risultati (default: 'output')
- `--show`: Mostra la visualizzazione dei risultati
- `--debug`: Abilita la modalità debug con visualizzazioni aggiuntive

## Esempio di Output

```
Debug mode enabled. Debug images will be saved to C:\Users\GabrielePanciotti\Desktop\Programming\FlyCatcher\FlyCatcher-Task\debug
Loaded image with shape: (3261, 4905, 3)
Punta 1 (area: 26282.5) unita alla matita 0 (colore: green)
Punta 2 (area: 26114.5) unita alla matita 1 (colore: green)
Rilevate 2 matite green
Punta 199 (area: 10115.5) unita alla matita 0 (colore: red)
Punta 198 (area: 5013.5) unita alla matita 1 (colore: red)
Rilevate 2 matite red
Punta 16 (area: 12294.5) unita alla matita 0 (colore: blue)
Rilevate 1 matite blue
Punta 24 (area: 24258.5) unita alla matita 0 (colore: yellow)
Rilevate 1 matite yellow
Punta 208 (area: 516.0) unita alla matita 0 (colore: purple)
Rilevate 1 matite purple
Punta 9 (area: 3335.5) unita alla matita 0 (colore: orange)
Rilevate 1 matite orange
Punta 184 (area: 31880.0) unita alla matita 0 (colore: pink)
Punta 184 (area: 31880.0) unita alla matita 1 (colore: pink)
Rilevate 2 matite pink
Rilevate 0 matite brown
Punta 14 (area: 25358.5) unita alla matita 0 (colore: light_blue)
Rilevate 1 matite light_blue
Punta 268 (area: 9273.0) unita alla matita 0 (colore: black)
Rilevate 1 matite black

Risultati salvati in results
Immagine risultato: results\pencils_detected.jpg
```
