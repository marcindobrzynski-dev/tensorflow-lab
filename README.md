# TensorFlow Lab

A hands-on learning repository exploring AI/ML fundamentals with TensorFlow and Keras - from basic linear regression to transfer learning with pretrained models.

## Table of Contents

- [Project Description](#project-description)
- [Tech Stack](#tech-stack)
- [Getting Started Locally](#getting-started-locally)
- [Available Scripts](#available-scripts)
- [Project Scope](#project-scope)
- [Project Status](#project-status)
- [License](#license)

## Project Description

This repository is a personal learning lab for building and experimenting with AI/ML models using TensorFlow. It contains progressive exercises that cover core machine learning concepts, starting from a simple linear regression model and advancing through convolutional neural networks (CNNs), image classification, and transfer learning.

Each script is a self-contained exercise that can be run independently.

## Tech Stack

- **Language:** Python 3
- **ML Framework:** [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- **Numerical Computing:** [NumPy](https://numpy.org/)
- **Runtime (some scripts):** [Google Colab](https://colab.research.google.com/)

### Datasets Used

| Dataset | Source | Used In |
|---|---|---|
| Fashion MNIST | Built into Keras | `basics-computer-vision.py`, `cnn-computer-vision_v01.py` |
| Horse or Human | [Google Storage](https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip) | `basics-horse-or-human_v01.py`, `basics-horse-or-human_v02.py`, `transfer-learning_v01.py` |
| ImageNet (pretrained weights) | Keras Applications | `transfer-learning_v01.py` |

## Getting Started Locally

### Prerequisites

- Python 3.10+
- pip

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/marcindobrzynski-dev/tensorflow-lab.git
   cd tensorflow-lab
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install tensorflow tensorflow-cpu numpy
   ```

> **Note:** Some scripts (`basics-horse-or-human_v02.py`, `transfer-learning_v01.py`) use `google.colab` for file uploads and are designed to run in [Google Colab](https://colab.research.google.com/). All other scripts run locally without modifications.

## Available Scripts

Run any script with:

```bash
python <script-name>.py
```

| Script | Description |
|---|---|
| `basics-model.py` | A minimal single-neuron model that learns the linear relationship `y = 2x - 1` using SGD. |
| `basics-computer-vision.py` | Fashion MNIST classification with a Dense neural network and a custom callback that stops training at 95% accuracy. |
| `cnn-computer-vision_v01.py` | Fashion MNIST classification improved with convolutional layers (Conv2D + MaxPooling). |
| `basics-horse-or-human_v01.py` | Binary image classification (horse vs. human) using a multi-layer CNN with `ImageDataGenerator`. |
| `basics-horse-or-human_v02.py` | Extended version with validation dataset and image upload for prediction (Google Colab). |
| `transfer-learning_v01.py` | Transfer learning using a pretrained InceptionV3 model with data augmentation and a custom classification head (Google Colab). |

## Project Scope

This project covers the following ML topics in a progressive learning path:

1. **Linear Regression** - Building a single-neuron model to learn a simple function.
2. **Dense Neural Networks** - Multi-layer networks for image classification (Fashion MNIST).
3. **Convolutional Neural Networks (CNNs)** - Using Conv2D and MaxPooling layers to improve image classification.
4. **Binary Image Classification** - Training a CNN from scratch on custom image data (Horse or Human).
5. **Validation & Inference** - Evaluating models with validation sets and running predictions on new images.
6. **Transfer Learning** - Leveraging pretrained models (InceptionV3/ImageNet) with frozen layers, custom heads, dropout, and data augmentation.

## Project Status

**Active** - This is an ongoing learning repository. New exercises and experiments are added as learning progresses.

## License

This project does not currently include a license. All rights reserved.
