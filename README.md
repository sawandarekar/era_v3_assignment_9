# ERA V3 Assignment 9

This repository contains a PyTorch implementation of ResNet50 for MNIST classification with a modular code structure.

## Project Structure

```
era_v3_assignment_9/
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── src/
│ ├── main.py # Main training script
│ ├── model.py # ResNet50 model architecture
│ ├── train.py # Training loop implementation
│ ├── validate.py # Validation/testing functions
│ ├── dataloader.py # Data loading and transformations
│ ├── checkpoint.py # Model checkpointing utilities
│ └── model_summary.py # Model visualization utilities
```


## File Descriptions

- **main.py**: Entry point of the application. Handles device setup, model initialization, and training loop orchestration.
- **model.py**: Contains the ResNet50 model architecture adapted for MNIST.
- **train.py**: Implements the training loop and learning rate scheduling.
- **validate.py**: Contains functions for model validation and testing.
- **dataloader.py**: Handles data loading, transformations, and augmentation for MNIST dataset.
- **checkpoint.py**: Utilities for saving and loading model checkpoints.
- **model_summary.py**: Functions for model visualization and parameter summary.

## Setup and Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

To train the model:
```bash
python src/main.py
```

The script will:
1. Download the MNIST dataset
2. Initialize the ResNet50 model
3. Train for 20 epochs using SGD optimizer
4. Display training progress and validation results

## Model Details

- Architecture: ResNet50 (adapted for MNIST)
- Input size: 1x28x28 (MNIST format)
- Optimizer: SGD with momentum
- Learning rate scheduler: StepLR
- Batch size: 128 (GPU) / 64 (CPU)
- Training epochs: 20

## Hardware Requirements

The code supports:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback option)

The device will be automatically selected based on availability.
