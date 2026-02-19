# TensorFlow Lab

A hands-on learning repository exploring AI/ML fundamentals with TensorFlow and Keras - from basic linear regression to transfer learning and natural language processing.

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

Scripts are organized by ML technique into numbered folders that follow a progressive learning path. Each script is a self-contained exercise that can be run independently.

## Tech Stack

- **Language:** Python 3
- **ML Framework:** [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/)
- **Data Loading:** [TensorFlow Datasets](https://www.tensorflow.org/datasets)
- **Numerical Computing:** [NumPy](https://numpy.org/)
- **Text Processing:** [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- **Runtime (some scripts):** [Google Colab](https://colab.research.google.com/)

### Datasets Used

| Dataset | Source | Used In |
|---|---|---|
| Fashion MNIST | Built into Keras / TFDS | `02-dense-neural-networks/basics-computer-vision.py`, `02-dense-neural-networks/tf-fashion-mnist-datasets.py`, `03-convolutional-neural-networks/cnn-computer-vision_v01.py` |
| Horse or Human | [Google Storage](https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip) | `04-binary-image-classification/basics-horse-or-human_v01.py`, `04-binary-image-classification/basics-horse-or-human_v02.py`, `05-transfer-learning/transfer-learning_v01.py` |
| IMDB Reviews | TensorFlow Datasets | `06-nlp-text-processing/imdb-tokenizer_v01.py` |
| ImageNet (pretrained weights) | Keras Applications | `05-transfer-learning/transfer-learning_v01.py` |

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
   pip install tensorflow tensorflow-cpu tensorflow-datasets numpy beautifulsoup4
   ```

> **Note:** Some scripts (`04-binary-image-classification/basics-horse-or-human_v02.py`, `05-transfer-learning/transfer-learning_v01.py`) use `google.colab` for file uploads and are designed to run in [Google Colab](https://colab.research.google.com/). All other scripts run locally without modifications.

## Available Scripts

Run any script with:

```bash
python <folder>/<script-name>.py
```

### 01-linear-regression

| Script | Description |
|---|---|
| `basics-model.py` | A minimal single-neuron model that learns the linear relationship `y = 2x - 1` using SGD. |

### 02-dense-neural-networks

| Script | Description |
|---|---|
| `basics-computer-vision.py` | Fashion MNIST classification with a Dense neural network and a custom callback that stops training at 95% accuracy. |
| `basics-tf-datasets.py` | Loads Fashion MNIST via `tensorflow_datasets` and explores dataset structure (no model training). |
| `tf-fashion-mnist-datasets.py` | Fashion MNIST classification using `tensorflow_datasets` for data loading, normalized inputs, and a Dense network with Dropout. |

### 03-convolutional-neural-networks

| Script | Description |
|---|---|
| `cnn-computer-vision_v01.py` | Fashion MNIST classification improved with convolutional layers (Conv2D + MaxPooling). |

### 04-binary-image-classification

| Script | Description |
|---|---|
| `basics-horse-or-human_v01.py` | Binary image classification (horse vs. human) using a multi-layer CNN with `ImageDataGenerator`. |
| `basics-horse-or-human_v02.py` | Extended version with validation dataset and image upload for prediction (Google Colab). |

### 05-transfer-learning

| Script | Description |
|---|---|
| `transfer-learning_v01.py` | Transfer learning using a pretrained InceptionV3 model with data augmentation and a custom classification head (Google Colab). |

### 06-nlp-text-processing

| Script | Description |
|---|---|
| `first-steps-with-tokenization_v01.py` | Tokenizing Polish sentences with OOV handling and sequence padding using Keras Tokenizer. |
| `imdb-tokenizer_v01.py` | IMDB reviews preprocessing with BeautifulSoup HTML cleanup, stopword removal, and tokenization via TensorFlow Datasets. |

## Project Scope

This project covers the following ML topics in a progressive learning path:

1. **Linear Regression** - Building a single-neuron model to learn a simple function.
2. **Dense Neural Networks** - Multi-layer networks for image classification (Fashion MNIST), including data loading with TensorFlow Datasets.
3. **Convolutional Neural Networks (CNNs)** - Using Conv2D and MaxPooling layers to improve image classification.
4. **Binary Image Classification** - Training a CNN from scratch on custom image data (Horse or Human) with validation and inference.
5. **Transfer Learning** - Leveraging pretrained models (InceptionV3/ImageNet) with frozen layers, custom heads, dropout, and data augmentation.
6. **NLP Text Processing** - Tokenization, sequence padding, OOV handling, and text preprocessing with stopword removal.

## Project Status

**Active** - This is an ongoing learning repository. New exercises and experiments are added as learning progresses.

## License

This project does not currently include a license. All rights reserved.
