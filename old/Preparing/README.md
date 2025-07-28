# FlyCatcher Computer Vision Task

## Overview
This repository contains a modular and efficient framework for computer vision tasks focused on anomaly detection in logistics services like railways. The codebase is designed to be flexible, maintainable, and leverages state-of-the-art libraries for image processing, machine learning, and anomaly detection.

## Features
- Comprehensive image preprocessing pipeline
- Multiple anomaly detection algorithms
- Evaluation metrics and visualization tools
- Modular architecture for easy extension
- Efficient data handling for large datasets
- GPU acceleration support

## Project Structure
```
├── config/                  # Configuration files
├── data/                    # Data directory (not tracked by git)
│   ├── raw/                 # Raw input data
│   ├── processed/           # Processed data
│   └── output/              # Model outputs and results
├── models/                  # Model definitions and saved models
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data/                # Data loading and processing
│   ├── features/            # Feature extraction
│   ├── models/              # Model implementation
│   ├── visualization/       # Visualization utilities
│   └── utils/               # Utility functions
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Usage
The main entry points are in the `src` directory. Example usage:

```python
from src.data import dataset
from src.models import anomaly_detector

# Load and preprocess data
data_loader = dataset.DataLoader('data/raw')
preprocessed_data = data_loader.preprocess()

# Train anomaly detection model
model = anomaly_detector.AnomalyDetector()
model.train(preprocessed_data)

# Detect anomalies in new data
results = model.detect(new_data)
```

## License
MIT