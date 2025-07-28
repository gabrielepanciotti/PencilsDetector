# Pencil Counter and Localizer

Questo progetto implementa un sistema di computer vision tradizionale (non basato su AI) per analizzare un'immagine contenente matite colorate, con l'obiettivo di:

1. Contare le matite verdi
2. Localizzare le matite verdi nell'immagine
3. Contare e localizzare le altre matite, raggruppate per colore

## Requisiti

```
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
```

Installare i requisiti con:

```bash
pip install -r requirements.txt
```

## Utilizzo

### Esecuzione base

```bash
python main.py --image percorso/all/immagine/pencils.jpg
```

### Opzioni aggiuntive

```bash
python main.py --image percorso/all/immagine/pencils.jpg --output-dir output --show
```

Parametri:
- `--image`: Percorso all'immagine da analizzare (obbligatorio)
- `--output-dir`: Directory dove salvare i risultati (default: 'output')
- `--show`: Mostra l'immagine con i risultati dell'analisi

## Output

Il programma genera:

1. **Immagine annotata**: Visualizzazione dell'immagine originale con bounding box colorate attorno alle matite identificate, etichettate per colore
2. **File JSON**: Contiene i risultati dettagliati dell'analisi:
   - Numero totale di matite
   - Conteggio matite per colore
   - Posizione di ogni matita (coordinate del centro e bounding box)
   - Sezione specifica per le matite verdi

## Approccio tecnico

Il sistema utilizza tecniche di computer vision tradizionali:

1. **Preprocessing**:
   - Conversione dell'immagine in spazio colore HSV
   - Applicazione di filtri per ridurre il rumore

2. **Rilevamento matite verdi**:
   - Creazione di una maschera per isolare il colore verde
   - Identificazione dei contorni nella maschera
   - Filtraggio dei contorni in base a dimensione e forma

3. **Rilevamento di tutte le matite**:
   - Rilevamento dei bordi con Canny
   - Identificazione dei contorni
   - Filtraggio in base a dimensione e rapporto d'aspetto
   - Classificazione del colore in base ai valori HSV

4. **Visualizzazione**:
   - Disegno di bounding box colorate
   - Etichettatura delle matite per colore
   - Evidenziazione speciale per le matite verdi

## Struttura del codice

Il codice è organizzato in funzioni modulari:

- `load_image()`: Carica l'immagine dal percorso specificato
- `preprocess_image()`: Prepara l'immagine per l'analisi
- `create_mask_for_color()`: Crea una maschera binaria per un intervallo di colori
- `find_pencils_in_mask()`: Trova le matite in una maschera binaria
- `get_dominant_color()`: Estrae il colore dominante da una regione
- `classify_color()`: Classifica un colore in base ai suoi valori HSV
- `detect_green_pencils()`: Rileva specificamente le matite verdi
- `detect_all_pencils()`: Rileva tutte le matite e le classifica per colore
- `visualize_results()`: Crea una visualizzazione dei risultati
- `generate_results_json()`: Genera un file JSON con i risultati dettagliati

## Limitazioni e possibili miglioramenti

- La classificazione dei colori è basata su intervalli HSV predefiniti e potrebbe richiedere aggiustamenti per diverse condizioni di illuminazione
- Il rilevamento potrebbe essere migliorato con tecniche di segmentazione più avanzate
- L'implementazione attuale funziona meglio con matite ben separate e con colori distinti
