# Neural Network Lab (nn-lap)

Small project demonstrating a simple neural-network pipeline for classification using NumPy data and Keras/TensorFlow.

## Project overview

This repository contains scripts and assets to explore, preprocess, train and evaluate a basic neural network on a mushroom dataset (binary classification). The code is organized as small, focused scripts to show each step of a typical ML workflow.

## Contents

- `data/` - prepared NumPy arrays and the original `mushrooms.csv` dataset.
- `scripts/` - small scripts used for exploration, preprocessing, model definition and training:
	- `01_data_exploration.py` — exploratory data analysis.
	- `02_data_preprocessing.py` — preprocessing and saving train/test arrays.
	- `03_basic_nn.py` — example of a basic neural network model.
	- `04_keras_sequential.py` — Keras Sequential model definition.
	- `05_train_evaluate.py` — training and evaluation pipeline.
	- `utils.py` — helper functions used by scripts.
- `models/` - saved Keras models (`basic_mushroom_model.h5`, `best_model.h5`).
- `logs/` - TensorBoard logs (training and validation subfolders).
- `figures/` - saved figures from EDA or training.

## Requirements

Python dependencies are listed in `requirements.txt`. Recommended Python 3.8+.

Quick setup:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

This repo includes pre-saved NumPy datasets in `data/`:

- `X_train.npy`, `y_train.npy` — training set
- `X_test.npy`, `y_test.npy` — test set
- `mushrooms.csv` — original CSV used for exploration and preprocessing

If you need to recreate the NumPy arrays from the CSV, run:

```bash
python scripts/02_data_preprocessing.py
```

## Quickstart — train a model

Train and evaluate using the provided script:

```bash
python scripts/05_train_evaluate.py
```

This script will load the arrays from `data/`, train a Keras model, save checkpoints to `models/`, and write TensorBoard logs to `logs/`.

To inspect training progress with TensorBoard:

```bash
tensorboard --logdir logs --port 6006
```

Then open http://localhost:6006 in your browser.

## Using saved models

Saved models are available in `models/`. Example to load and predict:

```python
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('models/best_model.h5')
X = np.load('data/X_test.npy')
preds = model.predict(X)
```

## Reproducibility

- Use the included `requirements.txt` to pin dependencies.
- The `logs/` directory contains TensorBoard events for the original training runs.

## Notes and next steps

- The scripts are intentionally simple and educational; feel free to refactor into a package or add CLI flags.
- Consider adding a `Makefile` or CLI wrapper for common tasks.

---

If you want, I can:
- add badges (license, python version, build) to this README;
- expand the Quickstart with exact arguments used by training;
- create a small example Jupyter notebook demonstrating inference.

