# Pencil Detector

Un sistema di computer vision tradizionale per rilevare, contare e localizzare matite colorate nelle immagini.

## Caratteristiche

- Rilevamento di matite verdi utilizzando maschere di colore HSV
- Rilevamento di matite di tutti i colori utilizzando edge detection e classificazione del colore
- Filtri per area e aspect ratio per migliorare la precisione del rilevamento
- Visualizzazione dei risultati con bounding box colorate
- Esportazione dei risultati in formato JSON
- Modalità di debug per la visualizzazione di immagini intermedie

## Struttura del Progetto

```
pencil_detector/
├── main.py              # Funzioni principali di rilevamento
├── requirements.txt     # Dipendenze
├── README.md           # Documentazione
├── config.py           # Parametri di configurazione
├── run.py              # Script di esecuzione principale
│
├── image_utils/        # Utilità per il caricamento e preprocessing delle immagini
│   ├── __init__.py
│   ├── loader.py       # Caricamento immagini
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
Loaded image with shape: (3261, 4905, 3)
Found 2 green pencils
Found 5 pencils in total
Visualization saved to output/pencils_detected.jpg
Results saved to output/pencils_results.json

=== Pencil Detection Results ===
Total pencils found: 5
Green pencils found: 2

Green pencil positions:
  Green pencil 1: Center at (1234, 567), Bounding box: (1200, 500, 68, 134)
  Green pencil 2: Center at (2345, 678), Bounding box: (2300, 600, 90, 156)

Pencil counts by color:
  Blue: 1
  Green: 2
  Red: 1
  Yellow: 1

Results saved to output
```
