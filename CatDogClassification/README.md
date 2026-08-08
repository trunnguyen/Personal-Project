# Cat & Dog Classifier

A binary image classifier built with PyTorch, trained on the Kaggle Dogs vs Cats dataset, with a Tkinter desktop GUI for real-time inference.

---

## Demo

### Desktop GUI

The app lets you pick any image and get a real-time prediction.

![GUI demo](DEMO/GUI.png)

### Training Curves

20 epochs, early stopping with patience 5, SGD optimizer + ReduceLROnPlateau scheduler.

![Training and validation curves](DEMO/train_1.png)

### Sample Predictions

Latest sample predictions from the validation set. Correct predictions are shown in green and incorrect predictions in red.

![Sample predictions](DEMO/train_2.png)

---

## Results

| Metric | Value |
|---|---|
| Best validation accuracy | **78.00%** (epoch 20) |
| Best validation loss | **0.4477** |
| Final loss (best model) | **0.4574** |
| Training accuracy | **77.52%** |
| Model parameters | **16,783,938** |
| Training epochs | 20 (no early stop triggered) |
| Device | CPU |
| Classes | `cats` (0), `dogs` (1) — alphabetical via `ImageFolder` |

Training log excerpt:

```text
Epoch  1 | Train Loss: 0.6481 Acc: 61.97% | Val loss: 0.6296 Acc: 61.65%
Epoch 10 | Train Loss: 0.5216 Acc: 74.00% | Val loss: 0.5051 Acc: 75.20%
Epoch 20 | Train Loss: 0.4687 Acc: 77.52% | Val loss: 0.4477 Acc: 78.00%
Training complete - best val accuracy: 78.00%
```

---

## Project Structure

```text
cat_dog_classifier/
├── model.py                  # DogCatCNN architecture + load_model()
├── data.py                   # Transforms and get_dataloaders()
├── trainer.py                # train(), evaluate(), plot_history(), visualize_predictions()
├── train.py                  # Main entrypoint — run this to train
├── GUI.py                    # Tkinter desktop app for inference
├── GUI(ver1).py              # Earlier GUI version
├── log.py                    # Logger + history saver
├── Cat_Dog_Classification.py # Earlier standalone training script
├── best_model.pth            # Best trained model checkpoint
├── logs/                     # Training logs and history JSON files
├── DEMO/                     # Older demo screenshots
├── train_results_latest.png  # Latest training curves
├── sample_predictions_latest.png # Latest validation predictions
└── requirements.txt
```

## How to Run

**Train:**
```bash
python train.py
```

Expects data at:

```text
./data/train
./data/valid
```

with one subfolder per class.

**Run GUI:**
```bash
python GUI.py
```

Expects `best_model.pth` in the same directory.

## Architecture

- 2 convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool)
- Hidden FC layer (32768 → 512) with Dropout(0.5)
- Output FC layer (512 → 2)
- SGD optimizer
- ReduceLROnPlateau scheduler
- Data augmentation: random horizontal flip + rotation (train only)
- Early stopping with patience 5

## Data Processing

Images are resized to **128 × 128** and normalized using ImageNet-style channel statistics.

Training augmentation:

- Random horizontal flip
- Random rotation up to 20 degrees
- Resize to 128 × 128
- Conversion to tensor
- Normalization

Validation/inference preprocessing:

- Resize to 128 × 128
- Conversion to tensor
- Normalization

Expected dataset structure:

```text
data/
├── train/
│   ├── cats/
│   └── dogs/
└── valid/
    ├── cats/
    └── dogs/
```

`ImageFolder` assigns class indices alphabetically:

```text
cats → 0
dogs → 1
```

## Training Configuration

```text
Epochs:                 20
Batch size:             64
Optimizer:              SGD
Learning rate:          0.001
Loss function:          CrossEntropyLoss
Scheduler:              ReduceLROnPlateau
Scheduler factor:       0.5
Scheduler patience:     3
Early stopping patience: 5
Device:                 CPU
```

The best model is selected using the lowest validation loss and saved as:

```text
best_model.pth
```

Training history is saved as a JSON file in:

```text
logs/history_YYYYMMDD_HHMMSS.json
```

## A Bug Worth Knowing About

`ImageFolder` assigns class indices **alphabetically**, not in the order you list them. Since `cats` < `dogs` alphabetically, the training pipeline assigns:

```text
cats = 0
dogs = 1
```

The GUI must use the same order:

```python
CLASS_NAMES = ['Cat', 'Dog']  # cats=0, dogs=1 — matches ImageFolder order
```

This prevents the GUI from displaying the opposite class from the model's actual prediction.

## Requirements

```text
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
Pillow>=9.5.0
```

Install with:

```bash
pip install -r requirements.txt
```
