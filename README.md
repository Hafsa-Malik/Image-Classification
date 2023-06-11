# Image-Classification

This project focuses on image classification using machine learning and deep learning algorithms. The goal is to classify handwritten digits from the MNIST dataset. Two notebooks are provided, namely `Perceptron.ipynb` and `MulticlassLogisticRegression.ipynb`, which implements the perceptron and multiclass logistic regression algorithms for image classification, respectively.

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository `git clone https://github.com/Hafsa-Malik/Image-Classification.git` or directly download the zipped folder.

2. Open the notebooks:
- Start Google Colab or Jupyter Notebook.
- Navigate to the project directory where you cloned the repository.
- Open `Perceptron.ipynb` and `MulticlassLogisticRegression.ipynb` notebooks.
- You can also install the required libraries using the following commands to run them on local machine:
```
pip install numpy matplotlib tensorflow
pip install torch torchvision
```

## Dataset

The MNIST dataset is used in this project. It consists of a large set of 28x28 grayscale images of handwritten digits (0-9) along with their corresponding labels. The dataset is divided into a training set and a test set, with 60,000 and 10,000 images, respectively. The task is to train models that can accurately classify the handwritten digits based on the provided images.

## Perceptron.ipynb

The `Perceptron.ipynb` notebook focuses on implementing a binary perceptron for image classification. The following libraries are imported:

- `numpy` (as `np`) - A fundamental library for numerical operations in Python.
- `matplotlib.pyplot` (as `plt`) - A plotting library used for visualizations.
- `tensorflow` (as `tf`) - A popular deep learning framework used for building and training neural networks.

The image classification is done step-by-step, starting with loading the MNIST dataset using TensorFlow. The images are flattened and normalized to improve the model's performance. The perceptron model is implemented from scratch, trained on the training set, and its accuracy is evaluated on the test set. The notebook provides code explanations, visualizations, and discussions to help understand the perceptron algorithm and its performance in image classification.

## MulticlassLogisticRegression.ipynb

The `MulticlassLogisticRegression.ipynb` notebook focuses on implementing multiclass logistic regression for image classification. The following libraries are imported:

- `torch.utils.data.Dataset` - A PyTorch class for creating custom datasets.
- `torch.utils.data.DataLoader` - A PyTorch class for efficient data loading.
- `torchvision.transforms` - A module that provides common image transformations.
- `torchvision.datasets` - A module that provides access to popular datasets, including MNIST.
- `matplotlib.pyplot` (as `plt`) - A plotting library used for visualizations.

The notebook starts by defining a custom dataset using PyTorch's `Dataset` class, which handles loading and preprocessing the MNIST dataset. The images are transformed, normalized, and converted into tensors. Then, a data loader is created using `DataLoader` to efficiently load and iterate over the dataset during training. The multiclass logistic regression model is defined and trained on the MNIST dataset. The notebook provides code explanations, visualizations, and discussions to help understand the multiclass logistic regression algorithm and its performance in image classification.
