# FlyCatcher - Sistema Avanzato di Rilevamento Matite Colorate

Un sistema di computer vision avanzato per rilevare, contare e localizzare matite colorate nelle immagini, con particolare attenzione al rilevamento accurato delle punte e alla gestione di matite sovrapposte o vicine. Il sistema è progettato per funzionare con immagini di alta qualità di matite colorate disposte su una superficie piana, anche in condizioni di sovrapposizione parziale o vicinanza tra matite dello stesso colore.

## Caratteristiche Principali

- **Rilevamento Multi-Colore**: Supporto completo per matite di diversi colori

- **Divisione Intelligente**: 
  - Algoritmo di clustering K-means per separare matite dello stesso colore vicine o sovrapposte
  - Analisi dettagliata delle variazioni di tonalità (hue) all'interno di ogni bounding box
  - Verifica di sovrapposizione tra matite divise con calcolo IoU (Intersection over Union)
  - Filtro per evitare duplicati con soglia IoU > 0.2

- **Rilevamento Punte**: 
  - Sistema multi-fase per identificare le punte delle matite con maschere HSV specializzate
  - Algoritmo di associazione punta-base basato su tre criteri principali:
    1. Allineamento verticale (la punta deve essere sopra la base)
    2. Allineamento orizzontale (la punta deve essere centrata rispetto alla base)
    3. Dimensione relativa (la punta deve avere dimensioni proporzionate alla base)
  - Parametri personalizzati per ogni colore per ottimizzare l'associazione
  - Gestione speciale per punte difficili da rilevare (viola, rosa, nero)

- **Gestione Casi Speciali**: 
  - **Rosso**: Gestione del wrapping hue attorno a 180/0 gradi con doppia maschera
  - **Viola**: 
    - Range HSV estremamente ampio (115-175 per hue)
  - **Rosa**: 
    - Filtro per evitare falsi positivi

- **Visualizzazione Avanzata**: 
  - Output visivo con bounding box colorate secondo una mappa colori predefinita
  - Etichette per ogni matita con nome del colore e indice progressivo
  - Salvataggio dell'immagine risultante in alta risoluzione

- **Modalità Debug**: 
  - Generazione di oltre 100 immagini intermedie per analizzare ogni fase del processo
  - Salvataggio di maschere HSV per ogni colore
  - Visualizzazione delle regioni di interesse per ogni matita rilevata
  - Maschere separate per punte e basi
  - Log dettagliati delle operazioni di divisione e associazione
  - File CSV con informazioni su area, aspect ratio e altre metriche

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
