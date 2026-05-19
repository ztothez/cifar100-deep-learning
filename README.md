# CIFAR-100 CNN Experiments

Deep learning course assignment — convolutional neural network experiments on a 10-class subset of CIFAR-100. Final grade: **5/5**.

## Overview

This project trains CNNs from scratch on selected CIFAR-100 subsets and evaluates how model architecture, activation function, optimizer, batch size, and class selection affect classification accuracy. The same workflow was run across four different environments for comparison.

| Environment | Framework | Device |
|---|---|---|
| Local terminal | PyTorch | CUDA GPU |
| Local CPU | TensorFlow / Keras | CPU |
| Google Colab | PyTorch | Tesla T4 |
| AMD Developer Cloud | PyTorch (ROCm) | AMD Instinct MI300X |

## Key Results

| Experiment | Config | Best Accuracy |
|---|---|---|
| Baseline | ReLU, Adam, batch size 64 | 61.1% |
| Exp 1 | Deeper dense classifier | 60.5% |
| Exp 2 | 4 convolutional blocks | 58.2% |
| **Exp 3** | **LeakyReLU + Adam** | **71.9%** |
| Exp 4 | SGD with momentum | 57.6% |
| Exp 5a | LeakyReLU, batch size 128 | 66.2% |
| Exp 5b | LeakyReLU, batch size 32 | 71.2% |
| Dataset 2 | Animal classes, best config | 71.2% |

The strongest result was **71.9%** achieved by switching from ReLU to LeakyReLU with Adam optimizer. Simply adding more layers or a larger dense classifier did not improve performance — activation function choice had the largest single effect.

## Project Structure

```
cifar100-deep-learning/
├── main.py                        # PyTorch GPU script (local CUDA, animal classes)
├── CPU/
│   ├── cpu_main.py                # TensorFlow/Keras CPU script
│   └── amd_baseline.py            # TensorFlow baseline run on AMD cloud server
├── notebooks/
│   └── cifar-100-exercise.ipynb   # Full experiment notebook
├── reports/                       # PDF and MD result walkthroughs
├── results/                       # Training curves and prediction plots
├── requirements-pytorch.txt
├── requirements-tensorflow.txt
└── README.md
```

## Setup

**PyTorch (GPU/ROCm):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pytorch.txt
python main.py
```

**TensorFlow (CPU):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-tensorflow.txt
python CPU/cpu_main.py
```

## Dataset

CIFAR-100 is downloaded automatically via `torchvision` or `tensorflow.keras.datasets`. Two class subsets were used:

- **Dataset 1:** apple, aquarium_fish, baby, bear, beaver, bed, bee, beetle, bicycle, bottle (visually distinct)
- **Dataset 2:** bear, chimpanzee, crocodile, elephant, kangaroo, leopard, lion, tiger, wolf, seal (visually similar animals)

Dataset 2 consistently scored lower, confirming that class similarity makes the classification task harder regardless of model configuration.

## What I Learned

- LeakyReLU outperformed ReLU significantly by avoiding the dying neuron problem
- Deeper models underperformed on 32x32 images — spatial resolution becomes too small after multiple pooling layers
- Adam converged faster and more reliably than SGD for this dataset
- Smaller batch sizes (32) generalized better than larger ones (128)
- Class selection has a major impact on achievable accuracy
