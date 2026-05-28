# Digit Recognizer

Small MNIST digit recognizer with a Pygame drawing UI. The model is trained in TensorFlow and saved as `mnist_model.keras`, which the app loads to predict digits drawn on the canvas.

## Requirements

- Python 3.13
- TensorFlow, NumPy, Pygame

## Run

1. Activate the virtual environment in `digit-recog-env`.
2. Run the app:

```bash
python app.py
```

## Train

```bash
python model.py
```

## Explore data

```bash
python explore_data.py
```
